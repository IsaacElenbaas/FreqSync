#include <fcntl.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/tx.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#define WINDOW_MS 100
// how much of the previous window to keep when analyzing frequencies
// TODO: not accounted for with history_seconds yet
#define WINDOW_OVERLAP_MS 0
// what number of "notes" to split an octave into to find the offset of the traditional 12-note equal temperament
#define ALIGN_BUCKETS_TO 1000

int pass = 1;
int sample_rate;
int window_size;
// raising this by one decreases movement by 1/2
int n_harmonics_2_pow = 5;
int DCT_TYPE;
int data_size;
int int_data_size;
double max_freq_amp;
double bucket_offset;
// how many notes to fit frequencies to
int octave_steps = 12;
// max amount of movement in notes to allow
double note_deviation = 0.5;
// how long ago to consider when weighing note interval importance
int history_seconds = 1;
// doubled, half searching half narrowing
int sync_rounds = 100000;
double* note_adjustments;
size_t max_packet_len;
size_t last_max_num_packets;
size_t max_num_packets = 0;
AVPacket** packets = NULL;

int handle_stream(AVFormatContext* format_context, AVStream* stream);
void pass_one_begin();
void pass_one_freqs(double* freqs);
void pass_one_end();
int pass_two_begin();
void pass_two_freqs(double* freqs);
int pass_two_end();
int pass_three_begin(int nb_streams, AVCodecParameters* codec_parameters);
void pass_three_freqs(double* freqs, int channel, int channels);
int pass_three_save_packet(int stream_idx, int channels);
void pass_three_end();
int write_output(AVFormatContext* format_context);

/*{{{ int main(int argc, char* argv[])*/
int main(int argc, char* argv[]) {
	// TODO: check argc for filename
	AVFormatContext* format_context = NULL;
	if(avformat_open_input(&format_context, argv[1], NULL, NULL) != 0) {
		fprintf(stderr, "Failed to open file!\n"); return 1; }
	if(avformat_find_stream_info(format_context, NULL) < 0) {
		avformat_close_input(&format_context);
		fprintf(stderr, "Failed to get stream information!\n"); return 1; }
	bool has_audio = false;
	for(unsigned int i = 0; i < format_context->nb_streams; i++) {
		if(format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
			sample_rate = format_context->streams[i]->codecpar->sample_rate;
			window_size = (WINDOW_MS*sample_rate)/1000;
			// tx issue, not sure why
			window_size -= window_size%100;
			max_packet_len = 0;
			last_max_num_packets = max_num_packets;
			for(pass = 1; pass <= 3; pass++) {
				if(has_audio) {
					if(av_seek_frame(format_context, i, 0, AVSEEK_FLAG_ANY) < 0) {
						if(note_adjustments != NULL) free(note_adjustments);
						if(packets != NULL) free(packets);
						avformat_close_input(&format_context);
						fprintf(stderr, "Failed to rewind to process next stream!\n"); return 1; }
				}
				else has_audio = true;
				int ret = 0;
				switch(pass) {
					case 1:       pass_one_begin(); break;
					case 2: ret = pass_two_begin(); break;
				}
				if(ret != 0) {
					if(note_adjustments != NULL) free(note_adjustments);
					if(packets != NULL) free(packets);
					avformat_close_input(&format_context); return 1; }
				ret = handle_stream(format_context, format_context->streams[i]);
				if(ret != 0) {
					if(note_adjustments != NULL) free(note_adjustments);
					if(packets != NULL) free(packets);
					avformat_close_input(&format_context); return 1; }
				switch(pass) {
					case 1:       pass_one_end();   break;
					case 2: ret = pass_two_end();   break;
					case 3:       pass_three_end(); break;
				}
				if(ret != 0) {
					if(note_adjustments != NULL) free(note_adjustments);
					if(packets != NULL) free(packets);
					avformat_close_input(&format_context); return 1; }
			}
		}
	}
	if(avformat_seek_file(format_context, -1, 0, 0, 0, AVSEEK_FLAG_ANY) < 0) {
		avformat_close_input(&format_context);
		fprintf(stderr, "Failed to rewind to write output!\n"); return 1; }
	write_output(format_context);
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
	if(codec_context == NULL) {
		fprintf(stderr, "Failed to allocate codec context!\n"); return 1; }
	if(avcodec_parameters_to_context(codec_context, stream->codecpar) < 0) {
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to copy codec parameters to codec context!\n"); return 1; }
	if(pass == 3 && pass_three_begin(format_context->nb_streams, stream->codecpar) != 0) {
		avcodec_free_context(&codec_context); return 1; }
	if(avcodec_open2(codec_context, avcodec_find_decoder(codec_context->codec_id), NULL) < 0) {
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to open codec for decoding!\n"); return 1; }
	AVPacket* packet = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();
	if(packet == NULL || frame == NULL) {
		if(packet != NULL) av_packet_free(&packet);
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to allocate packet or frame!\n"); return 1; }
	AVTXContext* ctx = NULL;
	av_tx_fn tx_fn;
	float scale = 1;
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
	if(av_tx_init(&ctx, &tx_fn, DCT_TYPE, /*invert:*/ 0, window_size, &scale, 0) < 0) {
		av_tx_uninit(&ctx);
		av_frame_free(&frame);
		av_packet_free(&packet);
		avcodec_free_context(&codec_context);
		fprintf(stderr, "Failed to create DCT transaction!\n"); return 1; }
	int channels = codec_context->ch_layout.nb_channels;
	data_size = av_get_bytes_per_sample(codec_context->sample_fmt);
	int_data_size = data_size;
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
	for(int channel = 1; channel < channels; channel++) {
		channel_windows[channel] = (void*)(((uintptr_t)channel_windows[channel-1]+window_size*data_size+(64-1)) & ~((uintptr_t)64-1));
	}
	/*}}}*/

	/*{{{ main loop*/
	int window_progress = 0;
	size_t num_packets = 0;
	while(av_read_frame(format_context, packet) == 0) {
		num_packets++;
		if(packet->stream_index == stream->index) {
			if(avcodec_send_packet(codec_context, packet) < 0) {
				fprintf(stderr, "Failed to send an audio packet!\n"); ret = 1; break; }
			int deferred_samples = 0;
			size_t packet_len = 0;
			while(ret == 0) {
				if(deferred_samples == 0) {
					ret = avcodec_receive_frame(codec_context, frame);
					if(ret == 0 && pass == 1) packet_len += channels*frame->nb_samples;
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
					for(int channel = 0; channel < channels; channel++) {
						if(int_data_size == data_size) {
							if(av_sample_fmt_is_planar(codec_context->sample_fmt)) {
								memcpy(
									(void*)((uintptr_t)channel_windows[channel]+window_progress*data_size),
									frame->data[channel]+(frame->nb_samples-nb_samples)*data_size,
									(nb_samples-deferred_samples)*data_size
								);
							}
							else {
								for(int i = 0; i < (nb_samples-deferred_samples); i++) {
									memcpy(
										(void*)((uintptr_t)channel_windows[channel]+(window_progress+i)*data_size),
										frame->data[0]+(frame->nb_samples-nb_samples+i*channels+channel)*data_size,
										data_size
									);
								}
							}
						}
						else {
							for(int i = 0; i < nb_samples-deferred_samples; i++) {
								uint64_t value = 0;
								if(av_sample_fmt_is_planar(codec_context->sample_fmt))
									memcpy(&value, frame->data[channel]+(frame->nb_samples-nb_samples+i)*int_data_size, int_data_size);
								else
									memcpy(&value, frame->data[0]+(frame->nb_samples-nb_samples+i*channels+channel)*int_data_size, int_data_size);
								int little_endian = 1;
								if(*(char*)&little_endian == 1) value = value << 8*(8-int_data_size);
								*(uint32_t*)((uintptr_t)channel_windows[channel]+(window_progress+i)*data_size) = value >> 32;
							}
						}
					}
					window_progress += nb_samples-deferred_samples;
				}
				// TODO: right now losing the last bit of the audio, not sure if I want to change that as frequency detection won't work
				if(window_progress == window_size) {
					// first output is average signal strength, then first harmonic, second harmonic, third, fourth, so on
					// which means that the frequency is k/2 times sample_rate/samples_slash_buckets
					// which is intuitive because if you sampled a second, the right things cancel and so yeah index 1 is first harmonic (just one centered up and down motion) which takes 2s to get back to its start, frequency of 0.5hz
					memset(freqs, 0, window_size*sizeof(double));
					for(int channel = 0; channel < channels; channel++) {
						tx_fn(ctx, out, channel_windows[channel], data_size);
						if(DCT_TYPE == AV_TX_FLOAT_DCT) { for(int i = 0; i < window_size; i++) {
								double value = ((float*)out)[i];
								freqs[i] += (pass != 3) ? fabs(value/channels) : value;
						} }
						else if(DCT_TYPE == AV_TX_DOUBLE_DCT) { for(int i = 0; i < window_size; i++) {
								double value = ((double*)out)[i];
								freqs[i] += (pass != 3) ? fabs(value/channels) : value;
						} }
						else {
							double max = (((uint64_t)1) << 32)-1;
							for(int i = 0; i < window_size; i++) {
								double value = ((uint32_t*)out)[i]/max;
								freqs[i] += (pass != 3) ? fabs(value/channels) : value;
							}
						}
						if(pass == 3) {
							pass_three_freqs(freqs, channel, channels);
							memset(freqs, 0, window_size*sizeof(double));
						}
					}
					switch(pass) {
						case 1: pass_one_freqs(freqs); break;
						case 2: pass_two_freqs(freqs); break;
					}
					if(pass != 3) {
						for(int channel = 0; channel < channels; channel++) {
							memmove(channel_windows[channel], (void*)((uintptr_t)channel_windows[channel]+(window_size-WINDOW_OVERLAP_MS)*data_size), WINDOW_OVERLAP_MS*data_size);
						}
					}
					window_progress = (pass != 3) ? WINDOW_OVERLAP_MS : 0;
				}
			}
			if(ret == 0 || ret == AVERROR_EOF) {
				if(pass == 1) {
					if(packet_len < (unsigned int)(channels*window_size)) packet_len = channels*window_size;
					if(packet_len > max_packet_len) max_packet_len = packet_len;
				}
				if(pass == 3 && pass_three_save_packet(stream->index, channels) != 0) ret = 1;
			}
			if(ret != 0) break;
		}
	}
	if(num_packets > max_num_packets) max_num_packets = num_packets;
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
	max_freq_amp = 0;
}
void pass_one_freqs(double* freqs) {
	for(int i = 1; i < window_size; i++) {
		double log_freq = log(((double)i)/2*((double)sample_rate)/window_size)/log(2);
		log_freq -= floor(log_freq);
		int index = floor(log_freq/(1/(double)ALIGN_BUCKETS_TO));
		bucket_candidates[index] = freqs[i]/(window_count+1)+bucket_candidates[index]*window_count/(double)(window_count+1);
		if(freqs[i] > max_freq_amp) max_freq_amp = freqs[i];
	}
	window_count++;
}
void pass_one_end() {
	double max_freq_avg_amp = -1;
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
int freq_combo_count;
double* freq_combo_weights;
int freq_combo_weight_count;

	/*{{{ int pass_two_begin()*/
int pass_two_begin() {
	int freq_combo_count = 1;
	for(int i = 2; i <= octave_steps-2; i++) { freq_combo_count *= i; }
	freq_combo_count = freq_combo_count*(octave_steps-1)*octave_steps/(2*freq_combo_count);
	freq_history_count = 0;
	freq_history_windows = history_seconds*(sample_rate+window_size-1)/window_size;
	freq_avg = malloc((octave_steps+freq_combo_count)*sizeof(double));
	if(freq_avg == NULL) {
		fprintf(stderr, "Failed to allocate memory!\n"); return 1; }
	memset(freq_avg, 0, (octave_steps+freq_combo_count)*sizeof(double));
	freq_combo_weights = (double*)((uintptr_t)freq_avg+octave_steps*sizeof(double));
	return 0;
}
	/*}}}*/

	/*{{{ void pass_two_freqs(double* freqs)*/
void pass_two_freqs(double* freqs) {
	for(int i = 0; i < octave_steps; i++) {
		freq_avg[i] *= freq_history_count/(double)(freq_history_count+1);
	}
	for(int i = 1; i < window_size; i++) {
		double log_freq = log(((double)i)/2*((double)sample_rate)/window_size)/log(2);
		log_freq -= bucket_offset;
		// center buckets on the note
		log_freq += 0.5/octave_steps;
		log_freq -= (log_freq >= 0) ? floor(log_freq) : ceil(log_freq);
		int index = log_freq/(1/(double)octave_steps);
		// having a counter for this is annoying. . . just trusting it to be near what it should be on average
		// this is probably why I can't get loudness to quite match up with what it was in the input
		freq_avg[index] += (freqs[i]/max_freq_amp)/(window_size/(double)octave_steps)/(freq_history_count+1);
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
	/*}}}*/

	/*{{{ int pass_two_end()*/
int pass_two_end() {

		/*{{{ remove noise from combo weights and normalize*/
	double min_combo_weight = 1;
	double max_combo_weight = 0;
	int counter = 0;
	for(int i = 0; i < octave_steps; i++) {
		for(int j = i+1; j < octave_steps; j++, counter++) {
			if(freq_combo_weights[counter] < min_combo_weight)
				min_combo_weight = freq_combo_weights[counter];
			if(freq_combo_weights[counter] > max_combo_weight)
				max_combo_weight = freq_combo_weights[counter];
		}
	}
	counter = 0;
	for(int i = 0; i < octave_steps; i++) {
		for(int j = i+1; j < octave_steps; j++, counter++) {
			freq_combo_weights[counter] = (freq_combo_weights[counter]-min_combo_weight)/(max_combo_weight-min_combo_weight);
		}
	}
		/*}}}*/

		/*{{{ initialize note_adjustments and create last_note_adjustments*/
	note_adjustments = malloc(octave_steps*sizeof(double));
	double* last_note_adjustments = malloc(octave_steps*sizeof(double));
	if(note_adjustments == NULL || last_note_adjustments == NULL) {
		if(note_adjustments != NULL) free(note_adjustments);
		fprintf(stderr, "Failed to allocate memory!\n"); return 1; }
	memset(note_adjustments, 0, octave_steps*sizeof(double));
	memset(last_note_adjustments, 0, octave_steps*sizeof(double));
		/*}}}*/

	int n_harmonics = pow(2, n_harmonics_2_pow-1)+1;
	double* harmonics = malloc(n_harmonics*sizeof(double));
	if(harmonics == NULL) {
		free(note_adjustments); free(last_note_adjustments);
		fprintf(stderr, "Failed to allocate memory!\n"); return 1; }

		/*{{{ fill harmonics*/
	memset(harmonics, 0, n_harmonics*sizeof(double));
	// O(n^2) but I don't want to write horrible garbage to make it be linear
	// shouldn't really ever be an issue, very small scale
	for(int i = n_harmonics-2; i >= 0; i--) {
		double harmonic = log(2*i+1)/log(2);
		harmonic -= floor(harmonic);
		int j = 0;
		if(i != 0) {
			while(harmonics[j] != 0 && harmonics[j] < harmonic) { j++; }
		}
		do {
			double temp = harmonics[j];
			harmonics[j] = harmonic;
			harmonic = temp;
			j++;
		} while(harmonic != 0);
	}
	harmonics[n_harmonics-1] = 1;
		/*}}}*/

	for(int round = 0; round < 2*sync_rounds; round++) {
		for(int i = 0; i < octave_steps; i++) {
			last_note_adjustments[i] = note_adjustments[i];
		}
		for(int i = 0; i < octave_steps; i++) {
			double i_log_freq = i/(double)octave_steps+last_note_adjustments[i];
			double sum_freq_combo_weights = 0;
			double left = 0, right = 0;
			for(int j = 0; j < octave_steps; j++) {
				if(j == i) continue;
				double freq_combo_weight = (j > i)
					? freq_combo_weights[(octave_steps*i-(i*i+i)/2)+(j-i-1)]
					: freq_combo_weights[(octave_steps*j-(j*j+j)/2)+(i-j-1)];
				sum_freq_combo_weights += freq_combo_weight;
				double interval = j/(double)octave_steps+last_note_adjustments[j];
				interval = (interval > i_log_freq) ? interval-i_log_freq : i_log_freq-interval;
				int k = 1;
				for(; k < n_harmonics-1; k++) {
					if(harmonics[k] > interval) break;
				}
				left  += freq_combo_weight*(interval-harmonics[k-1])/(octave_steps-1);
				right += freq_combo_weight*(harmonics[k]-interval)  /(octave_steps-1);
			}
			double shift = (left > right) ? right : -left;
			double percentage = (round < sync_rounds) ? 0.5 : pow(0.5, (round-sync_rounds+1)+1);
			// divide by 2 because others may move in the opposite direction
			// negative because we are moving the lower note but left/right are based on how the right should move
			note_adjustments[i] = last_note_adjustments[i]-percentage*(shift/sum_freq_combo_weights/2);
			if(fabs(note_adjustments[i]) > note_deviation/octave_steps) note_adjustments[i] = copysign(note_deviation/octave_steps, note_adjustments[i]);
		}
	}
	double average = 0;
	for(int i = 0; i < octave_steps; i++) { average += note_adjustments[i]/octave_steps; }
	for(int i = 0; i < octave_steps; i++) { note_adjustments[i] -= average; }
	for(int i = 0; i < octave_steps; i++) {
		printf("%d %lf\n", i, note_adjustments[i]/(1/(double)octave_steps));
	}
	free(harmonics);
	free(last_note_adjustments);
	free(freq_avg);
	return 0;
}
	/*}}}*/
/*}}}*/

/*{{{ pass three*/
	/*{{{ int pass_three_begin(int nb_streams, AVCodecParameters* codec_parameters)*/
AVCodecContext* out_codec_context;
AVFrame* out_frame;
int out_frame_progress;
ssize_t stream_packet_count;
double* packet_data;
void *packet_temp, *packet_temp_2;
float out_scale;
AVTXContext* out_ctx = NULL;
av_tx_fn out_tx_fn;
int pass_three_begin(int nb_streams, AVCodecParameters* codec_parameters) {
	out_codec_context = avcodec_alloc_context3(NULL);
	if(out_codec_context == NULL) {
		fprintf(stderr, "Failed to allocate encoding context!\n"); return 1; }
	if(avcodec_parameters_to_context(out_codec_context, codec_parameters) < 0) {
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to copy codec parameters to output codec context!\n"); return 1; }
	if(avcodec_open2(out_codec_context, avcodec_find_encoder(out_codec_context->codec_id), NULL) < 0) {
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to open codec for encoding!\n"); return 1; }
	out_frame = av_frame_alloc();
	if(out_frame == NULL) {
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to allocate a frame!\n"); return 1; }
	out_frame->format = out_codec_context->sample_fmt;
	out_frame->ch_layout = out_codec_context->ch_layout;
	out_frame->sample_rate = sample_rate;
	out_frame->nb_samples = out_codec_context->frame_size;
	if(av_frame_get_buffer(out_frame, 0) < 0) {
		av_frame_free(&out_frame);
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to allocate frame buffer!\n"); return 1; }
	out_frame_progress = 0;
	// 2x for packet_temp_2 because DCT-3 gives imaginaries
	packet_data = malloc(max_packet_len*sizeof(double)+(window_size+2)*data_size+(64-1)+2*window_size*data_size+(64-1));
	if(packet_data == NULL) {
		av_frame_free(&out_frame);
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to allocate temporary space for a packet!\n"); return 1; }
	packet_temp = (void*)(((uintptr_t)packet_data+max_packet_len*sizeof(double)+(64-1)) & ~((uintptr_t)64-1));
	memset((void*)((uintptr_t)packet_temp+window_size*data_size), 0, 2*data_size);
	packet_temp_2 = (void*)(((uintptr_t)packet_temp+(window_size+2)*data_size+(64-1)) & ~((uintptr_t)64-1));
	if(packets == NULL) {
		packets = malloc((nb_streams*max_num_packets+1)*sizeof(AVPacket*));
		memset(packets, 0, nb_streams*max_num_packets*sizeof(AVPacket*));
	}
	else {
		packets = realloc(packets, (nb_streams*max_num_packets+1)*sizeof(AVPacket*));
		if(max_num_packets != last_max_num_packets) {
			size_t new = nb_streams*(max_num_packets-last_max_num_packets);
			memset(&packets[nb_streams*max_num_packets-new], 0, new*sizeof(AVPacket*));
			for(int i = nb_streams-1; i >= 0; i--) {
				memmove(&packets[i*max_num_packets*sizeof(AVPacket*)], &packets[i*last_max_num_packets*sizeof(AVPacket*)], last_max_num_packets*sizeof(AVPacket*));
			}
		}
	}
	if(packets == NULL) {
		free(packet_data);
		av_frame_free(&out_frame);
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to allocate packets buffer!\n"); return 1; }
	out_scale = 0.5/(float)window_size;
	if(av_tx_init(&out_ctx, &out_tx_fn, DCT_TYPE, /*invert:*/ 1, window_size, &out_scale, 0) < 0) {
		free(packets); free(packet_data);
		av_frame_free(&out_frame);
		avcodec_free_context(&out_codec_context);
		fprintf(stderr, "Failed to create output DCT transaction!\n"); return 1; }
	return 0;
}
	/*}}}*/

	/*{{{ packet saving*/
size_t packet_window_count = 0;
void pass_three_freqs(double* freqs, int channel, int channels) {
	double* out_freqs = &packet_data[channel*(max_packet_len/channels)+packet_window_count*window_size];
	memset(out_freqs, 0, window_size*sizeof(double));
	for(int i = 1; i < window_size; i++) {
		double base_log_freq = log(((double)i)/2*((double)sample_rate)/window_size)/log(2);
		double log_freq = base_log_freq-bucket_offset;
		// center buckets on the note
		log_freq += 0.5/octave_steps;
		log_freq -= (log_freq >= 0) ? floor(log_freq) : ceil(log_freq);
		double index = log_freq/(1/(double)octave_steps);
		double adjust = (index-floor(index))*note_adjustments[(int)index]+(1-(index-floor(index)))*note_adjustments[((int)index+1)%octave_steps];
		int i2 = round(exp((base_log_freq+adjust)*log(2))*window_size/((double)sample_rate)*2);
		if(i2 < 0) i2 = 0;
		if(i2 >= window_size) i2 = window_size-1;
		out_freqs[i2] += freqs[i]/2;
	}
	if(channel == channels-1) packet_window_count++;
}
int pass_three_save_packet(int stream_idx, int channels) {
	if(packet_window_count == 0) { stream_packet_count++; return 0; }
	AVFrame* frame = av_frame_alloc();
	if(frame == NULL) {
		fprintf(stderr, "Failed to allocate a frame!\n"); return 1; }
	frame->format = out_codec_context->sample_fmt;
	frame->ch_layout = out_codec_context->ch_layout;
	frame->sample_rate = sample_rate;
	frame->nb_samples = window_size;
	if(av_frame_get_buffer(frame, 0) < 0) {
		av_frame_free(&frame);
		fprintf(stderr, "Failed to allocate frame buffer!\n"); return 1; }

	bool planar = av_sample_fmt_is_planar(out_codec_context->sample_fmt);
	for(size_t i = 0; i < packet_window_count; i++) {
		for(int channel = 0; channel < channels; channel++) {
			for(int j = 0; j < window_size; j++) {
				double value = packet_data[channel*(max_packet_len/channels)+i*window_size+j];
				if(DCT_TYPE == AV_TX_FLOAT_DCT)
					((float*)packet_temp)[j] = value;
				else if(DCT_TYPE == AV_TX_DOUBLE_DCT)
					((double*)packet_temp)[j] = value;
				else {
					uint64_t int_value = (((uint64_t)1) << (data_size*8))-1;
					int_value = value*int_value;
					int little_endian = 1;
					if(*(char*)&little_endian != 1) int_value = int_value << 8*(8-data_size);
					memcpy((void*)((uintptr_t)packet_temp+j*data_size), &int_value, data_size);
				}
			}
			out_tx_fn(out_ctx, packet_temp_2, packet_temp, data_size);
			for(int j = 1; j < window_size; j++) {
				memcpy((void*)((intptr_t)packet_temp_2+j*data_size), (void*)((intptr_t)packet_temp_2+2*j*data_size), data_size);
			}
			if(int_data_size == data_size) {
				if(planar) {
					memcpy(
						frame->data[channel],
						packet_temp_2,
						window_size*data_size
					);
				}
				else {
					for(int j = 0; j < window_size; j++) {
						memcpy(
							(void*)((uintptr_t)frame->data[0]+(j*channels+channel)*data_size),
							(void*)((uintptr_t)packet_temp_2+j*data_size),
							data_size
						);
					}
				}
			}
			else {
				for(int j = 0; j < window_size; j++) {
					uint64_t value = 0;
					memcpy(&value, (void*)((uintptr_t)packet_temp_2+j*data_size), data_size);
					int little_endian = 1;
					if(*(char*)&little_endian != 1) value = value >> 8*(8-int_data_size);
					memcpy((planar)
						? (void*)((uintptr_t)frame->data[channel]+j*data_size)
						: (void*)((uintptr_t)frame->data[0]+(j*channels+channel)*data_size),
						&value, int_data_size
					);
				}
			}
		}
		int remaining = window_size;
		while(remaining > 0) {
			int count = out_frame->nb_samples-out_frame_progress;
			if(count > remaining) count = remaining;
			if(planar) {
				for(int channel = 0; channel < channels; channel++) {
					memcpy(
						out_frame->data[channel]+out_frame_progress*int_data_size,
						frame->data[channel]+(window_size-remaining)*int_data_size,
						count*int_data_size
					);
				}
			}
			else {
				memcpy(
					out_frame->data[0]+out_frame_progress*channels*int_data_size,
					frame->data[0]+(window_size-remaining)*channels*int_data_size,
					count*channels*int_data_size
				);
			}
			if(out_frame_progress+count == out_frame->nb_samples) {
				int ret = avcodec_send_frame(out_codec_context, out_frame);
				if(ret < 0 && ret != AVERROR(EAGAIN)) {
					av_frame_free(&frame);
					fprintf(stderr, "Failed to encode frame!\n"); return 1; }
				packets[stream_idx*max_num_packets+stream_packet_count+1] = av_packet_alloc();
				if(packets[stream_idx*max_num_packets+stream_packet_count+1] == NULL) {
					av_frame_free(&frame);
					fprintf(stderr, "Failed to allocate a packet!\n"); return 1; }
				ret = avcodec_receive_packet(out_codec_context, packets[stream_idx*max_num_packets+stream_packet_count+1]);
				if(ret != 0) {
					av_packet_free(&packets[stream_idx*max_num_packets+stream_packet_count+1]);
					if(ret != AVERROR(EAGAIN)) {
						av_frame_free(&frame);
						fprintf(stderr, "Failed to encode packet!\n"); return 1; }
				}
				else {
					// not flushing at end either, already losing data because cutting to % frequency size so not caring for now
					AVPacket* temp;
					for(size_t j = stream_packet_count; true; j--) {
						temp = packets[stream_idx*max_num_packets+j];
						packets[stream_idx*max_num_packets+j] = packets[stream_idx*max_num_packets+stream_packet_count+1];
						packets[stream_idx*max_num_packets+stream_packet_count+1] = temp;
						if(packets[stream_idx*max_num_packets+stream_packet_count+1] == NULL) break;
						else if(j == 0) {
							av_packet_free(&packets[stream_idx*max_num_packets+stream_packet_count+1]);
							av_frame_free(&frame);
							fprintf(stderr, "Not enough buffer for packet!\n"); return 1; }
					}
				}
				out_frame_progress = 0;
			}
			else out_frame_progress += count;
			remaining -= count;
		}
	}

	av_frame_free(&frame);
	stream_packet_count++;
	packet_window_count = 0;
	return 0;
}
	/*}}}*/

void pass_three_end() {
	av_tx_uninit(&out_ctx);
	free(packet_data);
	av_frame_free(&out_frame);
	avcodec_free_context(&out_codec_context);
	free(note_adjustments);
}
/*}}}*/

/*{{{ int write_output(AVFormatContext* format_context)*/
int write_output(AVFormatContext* format_context) {

	/*{{{ setup*/
	int ret = 0;
	AVFormatContext* out_format_context = NULL;
	// TODO: change output filename - two places
	if(avformat_alloc_output_context2(&out_format_context, NULL, NULL, "out.mp3") < 0) {
		fprintf(stderr, "Failed to allocate output format context!\n"); return 1; }
	if(av_dict_copy(&out_format_context->metadata, format_context->metadata, 0) < 0) {
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to copy format context!\n"); return 1; }
	for(unsigned int i = 0; i < format_context->nb_streams; i++) {
		AVStream* output_stream = avformat_new_stream(out_format_context, NULL);
		if(output_stream == NULL) {
			avformat_free_context(out_format_context);
			fprintf(stderr, "Failed to create a stream!\n"); return 1; }
		if(avcodec_parameters_copy(output_stream->codecpar, format_context->streams[i]->codecpar) < 0) {
			avformat_free_context(out_format_context);
			fprintf(stderr, "Failed to copy stream parameters!\n"); return 1; }
	}
	if(avio_open(&out_format_context->pb, "out.mp3", AVIO_FLAG_WRITE) < 0) {
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to allocate packet!\n"); return 1; }
	AVPacket* packet = av_packet_alloc();
	if(packet == NULL) {
		avio_closep(&out_format_context->pb);
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to allocate packet!\n"); return 1; }
	size_t* stream_packet_counts = malloc(format_context->nb_streams*sizeof(size_t));
	if(stream_packet_counts == NULL) {
		av_packet_free(&packet);
		avio_closep(&out_format_context->pb);
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to allocate packet counts!\n"); return 1; }
	memset(stream_packet_counts, 0, format_context->nb_streams*sizeof(size_t));
	/*}}}*/

	if(avformat_write_header(out_format_context, NULL) < 0) {
		av_packet_free(&packet);
		avio_closep(&out_format_context->pb);
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to write header!\n"); return 1; }

	/*{{{ main loop*/
	while(av_read_frame(format_context, packet) == 0) {
		AVPacket* write_packet = packet;
		if(format_context->streams[packet->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
			write_packet = packets[packet->stream_index*max_num_packets+stream_packet_counts[packet->stream_index]];
			stream_packet_counts[packet->stream_index]++;
		}
		if(av_write_frame(out_format_context, write_packet) < 0) {
			av_packet_free(&packet);
			avio_closep(&out_format_context->pb);
			avformat_free_context(out_format_context);
			fprintf(stderr, "A write operation failed!\n"); return 1; }
	}
	/*}}}*/

	if(av_write_trailer(out_format_context) < 0) {
		free(stream_packet_counts);
		av_packet_free(&packet);
		avio_closep(&out_format_context->pb);
		avformat_free_context(out_format_context);
		fprintf(stderr, "Failed to write trailer!\n"); return 1; }

	/*{{{ cleanup*/
	av_packet_free(&packet);
	avio_closep(&out_format_context->pb);
	for(size_t i = 0; i < format_context->nb_streams*max_num_packets; i++) {
		if(packets[i] != NULL)
			av_packet_free(&packets[i]);
	}
	free(packets);
	avformat_free_context(out_format_context);
	return (ret == AVERROR_EOF) ? 0 : ret;
	/*}}}*/

}
/*}}}*/
