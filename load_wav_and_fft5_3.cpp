# include "Eigen/Dense"
# include "Eigen/Core"
# include "unsupported/Eigen/FFT"
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
# include <iostream>
# include <fstream>
# include <cmath>
/*
9/26 single channel version
multi channel FFT version



*/



std::vector<float> hanning(int frame_size) {
	std::vector<float> win(frame_size, 0);
	for (int i = 0; i < frame_size; i++) {
		float multiplier = 0.5 * (1 - cos(2 * M_PI * i / (frame_size - 1)));
		win[i] = multiplier;
	}
	return win;
}

std::vector<float> vector_float2float(std::vector<std::vector<float> > speech_array) {
	std::vector<float> result;
	for (int i = 0; i < speech_array.size(); i++) {
		std::vector<float> tmp = speech_array[i];
		for (int j = 0; j < tmp.size(); j++) {
			result.push_back(tmp[j]);
		}
	}
	return result;
}


void print_n() {
	std::cout << "\n" << std::endl;
}


int main(int argc, char** argv) {

	char* file_in = argv[1];
	char* file_out = argv[2];
	char* choose_index = argv[3];

	SNDFILE* sfr;
	SNDFILE* sfw;

	// structure including wav information
	SF_INFO sinfo;
	sinfo.format = 0;

	// analysis header file
	sfr = sf_open(file_in, SFM_READ, &sinfo);

	// output info.
	SF_INFO sinfo_out;
	sinfo_out = sinfo;
	sinfo_out.channels = 1;
	sfw = sf_open(file_out, SFM_WRITE, &sinfo_out);

	// ==================================
	// show wav header info.
	// ==================================
	std::cout << "==========================" << std::endl;
	std::cout << "wavinfo" << std::endl;
	std::cout << "==========================" << std::endl;
	printf("sinfo.frames   = %d  \n", sinfo.frames);
	printf("sinfo.format   = 0x%x\n", sinfo.format);
	printf("sinfo.channels = %d  \n", sinfo.channels);
	printf("sinfo.sections = %d  \n", sinfo.sections);
	printf("sinfo.seekable = %d  \n", sinfo.seekable);

	// ==================================
	// param. for frame processing 
	// ==================================		
	int FRAME_SHIFT = 160;
	int FRAME_SIZE = 400;
	int BUFFER_SIZE_INT = (int)FRAME_SHIFT / 2;
	int OVERLAP = FRAME_SIZE - FRAME_SHIFT;
	int NUMBER_OF_FRAME_BUFFER = (int)FRAME_SIZE / BUFFER_SIZE_INT;
	int GABAGE_FRAME = (int)FRAME_SHIFT / BUFFER_SIZE_INT;
	int NUMBER_OF_CHANNEL = sinfo.channels;
	int SELECTED_INDEX = atoi(choose_index);
	sf_count_t BUFFER_SIZE = (sf_count_t)BUFFER_SIZE_INT;
	std::vector<float> HANNING_WINDOW = hanning(FRAME_SIZE);

	// ==================================
	// init
	// ==================================
	Eigen::FFT<float> fft;
	std::ofstream outputfile("./test.txt");
	sf_count_t read_num;
	std::vector<float> buffer((int)BUFFER_SIZE * NUMBER_OF_CHANNEL, 0);// コンストラクタ引数は、(サイズ, 値)
	std::vector<std::vector<float> > stack_for_ola;
	std::vector<std::vector<float> > frame;
	std::vector<float> current_frame;
	// for FFT
	// complex spectrum
	std::vector <std::complex<float> > spectrum(FRAME_SIZE, 0);
	// single channel frame
	std::vector<float> single_frame(FRAME_SIZE, 0);
	int number_of_read_frame = 0;

	if (SELECTED_INDEX <= -1 || SELECTED_INDEX >= std::max(NUMBER_OF_CHANNEL - 1, 1)) {
		std::cout << "error! invalid channel" << std::endl;
		std::exit(1);
	}


	while (1) {

		std::cout << "==========================" << std::endl;
		std::cout << "read frame" << std::endl;
		std::cout << "==========================" << std::endl;

		std::cout << "reading # of frames:" << number_of_read_frame << std::endl;

		read_num = sf_readf_float(sfr, buffer.data(), (int)BUFFER_SIZE);
		if (read_num != (int)BUFFER_SIZE) {
			break;
		}

		// append until archieving frame size
		frame.push_back(buffer);

		std::cout << " appending:" << buffer.size() << " frame size:" << frame.size() << std::endl;
		print_n();

		// archieve frame size
		if (frame.size() == NUMBER_OF_FRAME_BUFFER) {
			std::cout << "==========================" << std::endl;
			std::cout << "Extract  N channel data" << std::endl;
			std::cout << "==========================" << std::endl;
			//extract n th channel waveform
			int frame_index = 0;
			std::cout << "frame size" << frame.size() << std::endl;
			std::cout << "single_frame size:" << single_frame.size() << std::endl;
			std::vector<float> reshape_frame = vector_float2float(frame);
			std::cout << "reshape_frame" << reshape_frame.size() << std::endl;

			int extract_sample = 0;
			for (int j = SELECTED_INDEX; j < FRAME_SIZE * NUMBER_OF_CHANNEL - (NUMBER_OF_CHANNEL - SELECTED_INDEX - 1); j += (NUMBER_OF_CHANNEL)) {
				single_frame[frame_index] = reshape_frame[j];
				frame_index = frame_index + 1;
				extract_sample = extract_sample + 1;
			}
			std::cout << "extract" << extract_sample << std::endl;
			std::cout << "frame_index" << frame_index << std::endl;
			std::cout << "spectrum size:" << spectrum.size() << " single_frame:" << single_frame.size() << std::endl;

			print_n();

			std::cout << "==========================" << std::endl;
			std::cout << "FFT" << std::endl;
			std::cout << "==========================" << std::endl;
			//FFT		  		  
			fft.fwd(spectrum, single_frame);

			// sum for test
			float sum_single_frame = 0;
			float sum_abs_spec = 0;
			for (int k = 0; k < FRAME_SIZE; k++) {
				sum_single_frame = sum_single_frame + (float)std::pow(std::abs(single_frame[k]), 2);
				sum_abs_spec = sum_abs_spec + (float)std::pow(std::abs(spectrum[k]), 2);
			}
			std::cout << " wavform pow:" << sum_single_frame << " spectrum pow:" << sum_abs_spec << std::endl;

			//write FFT's result
			for (int k = 0; k < FRAME_SIZE; k++) {
				outputfile << std::abs(spectrum[k]) << ",";
			}
			outputfile << "\n";


			print_n();

			std::cout << "==========================" << std::endl;
			std::cout << "something process" << std::endl;
			std::cout << "==========================" << std::endl;

			// apply window
			for (int s = 0; s < single_frame.size(); s++) {
				single_frame[s] = single_frame[s] * HANNING_WINDOW[s];
			}

			stack_for_ola.push_back(single_frame);

			std::cout << "push frame" << stack_for_ola.size() << std::endl;

			print_n();

			// =========================
			// OLA
			// =========================
			std::cout << "==========================" << std::endl;
			std::cout << "OLA" << std::endl;
			std::cout << "==========================" << std::endl;
			if (stack_for_ola.size() == 1) {
				current_frame = stack_for_ola[0];
				// original
				current_frame.erase(current_frame.begin() + FRAME_SHIFT, current_frame.begin() + current_frame.size());
			}

			else {
				std::vector<float> previous_frame = stack_for_ola[0];
				current_frame = stack_for_ola[1];

				int previous_frame_size = previous_frame.size();

				for (int i = 0; i < OVERLAP; i++) {
					current_frame[i] = current_frame[i] + previous_frame[previous_frame_size - (OVERLAP - 1) + i];
				}

				// original
				current_frame.erase(current_frame.begin() + FRAME_SHIFT, current_frame.begin() + current_frame.size());
				stack_for_ola.erase(stack_for_ola.begin());
			}

			std::cout << "current frame_size: " << current_frame.size() << std::endl;
			print_n();


			sf_writef_float(sfw, current_frame.data(), (int)BUFFER_SIZE);
			std::cout << "==========================" << std::endl;
			std::cout << "Delete frames" << std::endl;
			std::cout << "==========================" << std::endl;

			// for to next buffer
			std::cout << "delete frame size before:" << frame.size() << std::endl;
			for (int n = 0; n < GABAGE_FRAME - 1; n++) {  // -1
				frame.erase(frame.begin());
			}

			std::cout << "delete frame size after: " << frame.size() << std::endl;

			print_n();

		} // frame processing end

		number_of_read_frame = number_of_read_frame + 1;

	} // end of getting buffer

	// close file
	sf_close(sfr);
	sf_close(sfw);
	outputfile.close();

	return 0;

}