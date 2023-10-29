#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/tx.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define WINDOW_MS 100
// how much of the previous window to keep when analyzing frequencies
#define WINDOW_OVERLAP_MS 0
// what number of "notes" to split an octave into to find the offset of the traditional 12-note equal temperament
#define ALIGN_BUCKETS_TO 1000

int pass = 1;
int sample_rate;
int window_size;
double max_freq_avg_amp;
double bucket_offset;
int octave_steps = 12;
int history_seconds = 5;

int handle_stream(AVFormatContext* format_context, AVStream* stream);
void pass_one_begin();
void pass_one_freqs(double* freqs);
void pass_one_end();
// TODO: check for return value and bail
int pass_two_begin();
void pass_two_freqs(double* freqs);
void pass_two_end();

/*{{{ int main(int argc, char* argv[])*/
int main(int argc, char* argv[]) {
	AVFormatContext* format_context = NULL;
	if(avformat_open_input(&format_context, argv[1], NULL, NULL) != 0) {
		fprintf(stderr, "Failed to open file!\n"); return 1; }
	if(avformat_find_stream_info(format_context, NULL) < 0) {
		fprintf(stderr, "Failed to get stream information!\n");
		avformat_close_input(&format_context); return 1; }
	bool has_audio = false;
	for(unsigned int i = 0; i < format_context->nb_streams; i++) {
		if(format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
			sample_rate = format_context->streams[i]->codecpar->sample_rate;
			window_size = (WINDOW_MS*sample_rate)/1000;
			for(pass = 1; pass <= 2; pass++) {
				if(has_audio) {
					if(av_seek_frame(format_context, i, 0, AVSEEK_FLAG_BACKWARD) < 0) {
						fprintf(stderr, "Failed to rewind to process next stream!\n");
						avformat_close_input(&format_context); return 1; }
				}
				else has_audio = true;
				switch(pass) {
					case 1: pass_one_begin(); break;
					case 2: pass_two_begin(); break;
				}
				handle_stream(format_context, format_context->streams[i]);
				switch(pass) {
					case 1: pass_one_end(); break;
					case 2: pass_two_end(); break;
				}
			}
		}
	}
	avformat_close_input(&format_context);
	if(!has_audio) {
		fprintf(stderr, "Failed to find an audio stream!\n"); return 1; }
	return 0;
}
/*}}}*/

/*{{{ int handle_stream(AVFormatContext* format_context, AVStream* stream)*/
int handle_stream(AVFormatContext* format_context, AVStream* stream) {

	/*{{{ setup*/
	int ret = 0;
	AVCodecContext* codec_context = avcodec_alloc_context3(NULL);
	if(!codec_context) {
		fprintf(stderr, "Failed to allocate codec context!\n"); return 1; }
	if(avcodec_parameters_to_context(codec_context, stream->codecpar) < 0) {
		fprintf(stderr, "Failed to copy codec parameters to codec context!\n");
		avcodec_free_context(&codec_context);
		return 1;
	}
	if(avcodec_open2(codec_context, avcodec_find_decoder(codec_context->codec_id), NULL) < 0) {
		fprintf(stderr, "Failed to open codec!\n"); return 1; }
	AVPacket* packet = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();
	AVTXContext* ctx = NULL;
	av_tx_fn tx_fn;
	float scale = 1/(float)window_size;
	// TODO: need to set correct DCT data type
	// https://ffmpeg.org/doxygen/trunk/group__lavu__sampfmts.html#gaf9a51ca15301871723577c730b5865c5
	int DCT_TYPE;
	switch(stream->codecpar->format) {
		case AV_SAMPLE_FMT_FLT:
		case AV_SAMPLE_FMT_FLTP:
			DCT_TYPE = AV_TX_FLOAT_DCT; break;
		case AV_SAMPLE_FMT_DBL:
		case AV_SAMPLE_FMT_DBLP:
			DCT_TYPE = AV_TX_DOUBLE_DCT; break;
		default:
			DCT_TYPE = AV_TX_INT32_DCT;
	}
	if(av_tx_init(&ctx, &tx_fn, AV_TX_FLOAT_DCT, /*invert:*/ 0, window_size, &scale, 0) < 0) {
		av_tx_uninit(&ctx);
		av_frame_free(&frame);
		av_packet_free(&packet);
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to create DCT transaction!\n"); return 1; }
	int channels = codec_context->ch_layout.nb_channels;
	int data_size = av_get_bytes_per_sample(codec_context->sample_fmt);
	int int_data_size = data_size;
	if(DCT_TYPE == AV_TX_INT32_DCT && data_size != 32/8)
		data_size = 32/8;
	/*}}}*/

	/*{{{ DCT memory setup*/
	double* freqs = malloc(
		window_size*sizeof(double)+
		(1+channels)*window_size*data_size+
		(64-1)+
		channels*(64-(window_size*data_size)%64)
	);
	void* out = (void*)(((uintptr_t)freqs+window_size*sizeof(double)+(64-1)) & ~((uintptr_t)64-1));
	void** channel_windows = malloc(channels*sizeof(void*));
	if(freqs == NULL || channel_windows == NULL) {
		if(freqs != NULL) free(freqs);
		if(channel_windows != NULL) free(channel_windows);
		av_frame_free(&frame);
		av_packet_free(&packet);
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to allocate memory!\n"); return 1; }
	channel_windows[0] = (void*)(((uintptr_t)out+window_size*data_size+(64-1)) & ~((uintptr_t)64-1));
	for(int i = 1; i < channels; i++) {
		channel_windows[i] = (void*)(((uintptr_t)channel_windows[i-1]+window_size*data_size+(64-1)) & ~((uintptr_t)64-1));
	}
	/*}}}*/

	/*{{{ main loop*/
	int window_progress = 0;
	while(av_read_frame(format_context, packet) == 0) {
		if(packet->stream_index == stream->index) {
			if(avcodec_send_packet(codec_context, packet) < 0) {
				fprintf(stderr, "Failed to send an audio packet!\n"); ret = 1; break; }
			int deferred_samples = 0;
			while(ret == 0) {
				if(deferred_samples == 0) {
					ret = avcodec_receive_frame(codec_context, frame);
					deferred_samples = frame->nb_samples;
				}
				if(ret != AVERROR_EOF) {
					// didn't read enough data for the next frame
					if(ret == AVERROR(EAGAIN)) {
						ret = 0; break; }
					else if(ret < 0) {
						fprintf(stderr, "Failed to decode an audio packet!\n"); ret = 1; break; }
					else ret = 0;
					int nb_samples = deferred_samples;
					deferred_samples = (window_progress+nb_samples <= window_size)
						? 0
						: nb_samples-(window_size-window_progress);
					for(int i = 0; i < channels; i++) {
						if(int_data_size == data_size) {
							memcpy(
								(void*)((uintptr_t)channel_windows[i]+window_progress*data_size),
								frame->data[i]+(frame->nb_samples-nb_samples)*data_size,
								(nb_samples-deferred_samples)*data_size
							);
						}
						else {
							int little_endian = 1;
							for(int j = 0; j < nb_samples-deferred_samples; j++) {
								uint64_t value = 0;
								memcpy(&value, frame->data[i]+(frame->nb_samples-nb_samples+j)*int_data_size, int_data_size);
								if(*(char*)&little_endian == 1) value << 8*(8-int_data_size);
								*(uint32_t*)((uintptr_t)channel_windows[i]+(window_progress+j)*data_size) = value >> 32;
							}
						}
					}
					window_progress += nb_samples-deferred_samples;
				}
				if(window_progress == window_size) {
					// first output is average signal strength, then first harmonic, second harmonic, third, fourth, so on
					// which means that the frequency is k/2 times sample_rate/samples_slash_buckets
					// which is intuitive because if you sampled a second, the right things cancel and so yeah index 1 is first harmonic (just one centered up and down motion) which takes 2s to get back to its start, frequency of 0.5hz
					memset(freqs, 0, window_size*sizeof(double));
					for(int i = 0; i < channels; i++) {
						tx_fn(ctx, out, channel_windows[i], data_size);
						if(DCT_TYPE == AV_TX_FLOAT_DCT) { for(int j = 0; j < window_size; j++) {
								freqs[j] += fabsf(((float*)out)[j]/channels);
						} }
						else if(DCT_TYPE == AV_TX_DOUBLE_DCT) { for(int j = 0; j < window_size; j++) {
								freqs[j] += fabs(((double*)out)[j]/channels);
						} }
						else {
							double max = (((uint64_t)1) << 32)-1;
							for(int j = 0; j < window_size; j++) {
								freqs[j] += fabs((((uint32_t*)out)[j]/max)/channels);
							}
						}
					}
					switch(pass) {
						case 1: pass_one_freqs(freqs); break;
						case 2: pass_two_freqs(freqs); break;
					}
					// TODO: don't do on encoding pass
					for(int i = 0; i < channels; i++) {
						memcpy(channel_windows[i], (void*)((uintptr_t)channel_windows[i]+(window_size-WINDOW_OVERLAP_MS)*data_size), WINDOW_OVERLAP_MS*data_size);
					}
					window_progress = WINDOW_OVERLAP_MS;
				}
			}
			if(ret != 0) break;
		}
	}
	/*}}}*/

	/*{{{ cleanup*/
	free(channel_windows);
	free(freqs);
	av_tx_uninit(&ctx);
	av_frame_free(&frame);
	av_packet_free(&packet);
	avcodec_free_context(&codec_context);
	return (ret == AVERROR_EOF) ? 0 : ret;
	/*}}}*/

}
/*}}}*/

/*{{{ pass one*/
double bucket_candidates[ALIGN_BUCKETS_TO];
double window_count;
void pass_one_begin() {
	memset(bucket_candidates, 0, sizeof(bucket_candidates));
	window_count = 0;
}
void pass_one_freqs(double* freqs) {
	for(int i = 1; i < window_size; i++) {
		double log_freq = log(((double)i)/2*((double)sample_rate)/window_size)/log(2);
		log_freq -= floor(log_freq);
		int index = floor(log_freq/(1/(double)ALIGN_BUCKETS_TO));
		bucket_candidates[index] = window_count/(double)(window_count+1)*bucket_candidates[index]+freqs[i]/(window_count+1);
	}
	window_count++;
}
void pass_one_end() {
	max_freq_avg_amp = -1;
	for(int i = 0; i < ALIGN_BUCKETS_TO; i++) {
		if(bucket_candidates[i] > max_freq_avg_amp) {
			max_freq_avg_amp = bucket_candidates[i];
			bucket_offset = i/(double)ALIGN_BUCKETS_TO;
		}
	}
}
/*}}}*/

/*{{{ pass two*/
int freq_history_count;
int freq_history_windows;
double* freq_avg;
double* freq_combo_weights;
int freq_combo_weight_count;
int pass_two_begin() {
	int combos = 1;
	for(int i = 2; i <= octave_steps-2; i++) { combos *= i; }
	combos = combos*(octave_steps-1)*octave_steps/(2*combos);
	freq_history_count = 0;
	freq_history_windows = history_seconds*(sample_rate+window_size-1)/window_size;
	freq_avg = malloc((octave_steps+combos)*sizeof(double));
	if(freq_avg == NULL) {
		fprintf(stderr, "Failed to allocate memory!\n"); return 1; }
	memset(freq_avg, 0, (octave_steps+combos)*sizeof(double));
	freq_combo_weights = (double*)((uintptr_t)freq_avg+octave_steps);
	return 0;
}
void pass_two_freqs(double* freqs) {
	for(int i = 0; i < octave_steps; i++) {
		freq_avg[i] = freq_history_count/(double)(freq_history_count+1)*freq_avg[i];
	}
	for(int i = 1; i < window_size; i++) {
		double log_freq = log(((double)i)/2*((double)sample_rate)/window_size)/log(2);
		log_freq += bucket_offset;
		log_freq -= floor(log_freq);
		int index = log_freq/(1/(double)octave_steps);
		freq_avg[index] += (freqs[i]/max_freq_avg_amp)/(window_size/(double)octave_steps)/(freq_history_count+1);
	}
	if(freq_history_count+1 < freq_history_windows) freq_history_count++;
	int counter = 0;
	for(int i = 0; i < octave_steps; i++) {
		for(int j = i+1; j < octave_steps; j++, counter++) {
			freq_combo_weights[counter] = freq_combo_weight_count/(double)(freq_combo_weight_count+1)*freq_combo_weights[counter]+(freq_avg[i]*freq_avg[j])/(freq_combo_weight_count+1);
		}
	}
	freq_combo_weight_count++;
}
void pass_two_end() {
	free(freq_avg);
}
/*}}}*/
