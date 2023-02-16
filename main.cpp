#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::dnn;


int main() {

	VideoCapture cap(0);

	Mat frame;
	Mat frame_raw;

	Net net = readNet("model_0418_3.pb");

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
	}

	while (true) {

		cap >> frame;
		cap >> frame_raw;

		if (frame.empty())
			break;

		Mat image_hsv;
		cvtColor(frame, image_hsv, COLOR_BGR2HSV);

		vector<Mat> hsv_planes(3);
		split(image_hsv, hsv_planes);

		Mat V_plane = hsv_planes[2];

		float alpha = 50.f;
		Mat frame_new = V_plane + (V_plane - 200) * alpha;

		Mat bin;
		threshold(frame_new, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

		Mat labels, stats, centroids;
		int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);		

		int count = 0;
		vector<vector<int>> num_roi;
		//vector<int> num_roi;

		for (int i = 1; i < cnt; i++) {

			int* p = stats.ptr<int>(i);
			int pixel = frame.total() / 600;

			if (p[4] < 30) continue; //픽셀 개수가 20보다 작으면 무시
			if (p[4] > pixel) continue; //라벨링 영역의 픽셀 개수가 전체 이미지의 1/5 이상을 초과하면 무시			
			
			//if (p[0] - 3 <= 0 || p[1] - 3 <= 0 || p[0] + p[2] + 3 >= frame.rows || p[1] + p[3] + 6 >= frame.cols) continue;
			
			if (p[0] - 3 >= 0 && p[1] - 3 >= 0 && p[0] + p[2] + 3 <= frame.rows && p[1] + p[3] + 3 <= frame.cols) {
				
				Mat ROI_num;
				ROI_num = frame(Rect(p[0] - 3, p[1] - 3, p[2] + 6, p[3] + 6));
				//rectangle(frame, Rect(p[0] - 3, p[1] - 3, p[2] + 6, p[3] + 6), Scalar(0, 255, 255), 1);

				Mat ROI_num_resize;
				resize(ROI_num, ROI_num_resize, Size(224, 224), 0, 0, INTER_LINEAR);

				vector<Mat> bgr_planes;
				split(ROI_num_resize, bgr_planes);

				Mat BG_ROI_num(Size(224, 224), CV_8UC1);
				BG_ROI_num = 0.5*bgr_planes[0] + 0.5*bgr_planes[1];

				Mat blurred;
				int sigma = 3;
				GaussianBlur(BG_ROI_num, blurred, Size(), sigma);

				float alpha = 100.f;
				Mat dst(Size(224, 224), CV_8UC1);
				dst = (1 + alpha) * BG_ROI_num - alpha * blurred;

				Mat blob = blobFromImage(dst, 1 / 255.f, Size(224, 224));
				net.setInput(blob);
				Mat prob = net.forward();

				double maxVal;
				Point maxLoc;
				minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
				int digit = maxLoc.x;

				if (maxVal > 0.98) {
					vector<int> num;	

					if (digit == 1) { // 숫자 1은 width가 좁아서 평균 거리를 잴 때, 문제가 발생함. 이문제 해결을 위해 면적을 넓게 잡음
						num.push_back(p[0] - 4);
						num.push_back(p[1]);
						num.push_back(p[2]+8);
						num.push_back(p[3]);
						num.push_back(p[4]);
						num.push_back(digit);
					}
					else {
						for (int i = 0; i < 5; i++) {
							num.push_back(p[i]);
						}
						num.push_back(digit);
					}

					//rectangle(frame, Rect(num[0]-3, num[1]-3, num[2]+6, num[3] + 6), Scalar(0, 255, 255), 1);


					imshow("dst", dst);


					num_roi.push_back(num);

					count++;
				}	
			}		

		}		
		
		
		vector<vector<int>> num_roi_select; // 최종적으로 결정된 숫자 영역

		// 일직선상에 있는 숫자를 골라내는 부분
		int cnt_1 = 0;

		for (int i = 0; i < num_roi.size(); i++) {
			for (int j = 0; j < num_roi.size(); j++) {
				if (i == j) continue; // 같은 영역 비교 방지
				if (abs(num_roi[i][1] - num_roi[j][1]) <10 ) // 각 후보군에 대해서 다른 후보군과 비교 후 y좌표의 차이가 10 미만이면 count
					cnt_1++;
			}

			if (cnt_1 >= 3) {
				num_roi_select.push_back(num_roi[i]); // y좌표의 차이가 10 미만인 다른영역이 3개 이상 있으면 후보군에 편입
			}

			cnt_1 = 0;
		}
		num_roi.clear();
			 
		int height = 0;
		int height_avg;
		int width = 0;
		int width_avg;

		int row_size = num_roi_select.size();

		if (!num_roi_select.empty()) {
			for (int i = 0; i < num_roi_select.size(); i++) {
				height = height + num_roi_select[i][3];
				width = width + num_roi_select[i][2];
			}
			height_avg = height / num_roi_select.size();
			width_avg = width / num_roi_select.size();
		} 
			   		
		// Y축 좌표를 기준으로 IP1 과 IP2를 구분하기		
		
		vector<vector<int>> num_roi_IP1;
		vector<vector<int>> num_roi_IP2;

		if (!num_roi_select.empty()) {
			num_roi_IP1.push_back(num_roi_select[0]);

			for (int j = 1; j < num_roi_select.size(); j++) {
				if (abs(num_roi_select[0][1] - num_roi_select[j][1]) > 2 * height_avg) {
					num_roi_IP2.push_back(num_roi_select[j]);
				}
				else {
					num_roi_IP1.push_back(num_roi_select[j]);
				}
			}

			//IP1 수치를 화면에 출력

			for (int i = 0; i < num_roi_IP1.size(); i++) {
				rectangle(frame, Rect(num_roi_IP1[i][0] - 3, num_roi_IP1[i][1] - 3, num_roi_IP1[i][2] + 6, num_roi_IP1[i][3] + 6), Scalar(0, 255, 255), 1);

				int n1 = num_roi_IP1[i][5];
				char n2[10];
				sprintf_s(n2, "%d", n1);
				putText(frame, n2, Point(num_roi_IP1[i][0] - 3, num_roi_IP1[i][1] - 4), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255));
			}

			//IP2 수치를 화면에 출력

			for (int i = 0; i < num_roi_IP2.size(); i++) {
				rectangle(frame, Rect(num_roi_IP2[i][0] - 3, num_roi_IP2[i][1] - 3, num_roi_IP2[i][2] + 6, num_roi_IP2[i][3] + 6), Scalar(0, 255, 255), 1);

				int n1 = num_roi_IP2[i][5];
				char n2[10];
				sprintf_s(n2, "%d", n1);
				putText(frame, n2, Point(num_roi_IP2[i][0] - 3, num_roi_IP2[i][1] - 4), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255));
			}

		}
		num_roi_select.clear();		

		if (!num_roi_IP1.empty() & !num_roi_IP2.empty()) {
			cout << "ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ" << endl;
			cout << "Connected IP:" << " " << "2" << endl;
		}
		if((!num_roi_IP1.empty() & num_roi_IP2.empty()) || (num_roi_IP1.empty() & !num_roi_IP2.empty())) {
			cout << "ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ" << endl;
			cout << "Connected IP:" << " " << "1" << endl;
		}
		if (num_roi_IP1.empty() & num_roi_IP2.empty()) {
			cout << "ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ" << endl;
			cout << "Connected IP:" << " " << "0" << endl;
		}		
		

		//IP1 & IP2 수치를 x축 좌표를 기준으로 오름차순으로 정리
		vector<pair<int, int>> IP1;
		vector<pair<int, int>> IP1_Rate;
		vector<pair<int, int>> IP1_Volume;
		vector<pair<int, int>> IP2;
		vector<pair<int, int>> IP2_Rate;
		vector<pair<int, int>> IP2_Volume;

		if (!num_roi_IP1.empty() || !num_roi_IP2.empty()) {

			for (int i = 0; i < num_roi_IP1.size(); i++) {
				IP1.push_back(pair<int, int>(num_roi_IP1[i][0], num_roi_IP1[i][5]));
			}

			sort(IP1.begin(), IP1.end());
			num_roi_IP1.clear();

			//X좌표값을 기준으로 RATE 영역과 VOLUME 영역으로 구분하기 (IP1)
			int cnt_2 = 0;

			for (int i = 0; i < IP1.size() - 1; i++) {

				if (abs(IP1[i].first - IP1[i + 1].first) > 2 * width_avg) {
					break;
				}

				cnt_2++;

			}

			// VOLUME 수치의 자릿수에 따라 RATE와 VOLUME 영역으로 세밀하게 구분하기 (IP1)
			if (cnt_2 == 8) {
				for (int i = 0; i < 4; i++) {
					IP1_Rate.push_back(pair<int, int>(IP1[i].first, IP1[i].second));
				}
				for (int j = 4; j < IP1.size(); j++) {
					IP1_Volume.push_back(pair<int, int>(IP1[j].first, IP1[j].second));
				}
			}
			else if (cnt_2 == 7) {
				for (int i = 0; i < 3; i++) {
					IP1_Rate.push_back(pair<int, int>(IP1[i].first, IP1[i].second));
				}
				for (int j = 3; j < IP1.size(); j++) {
					IP1_Volume.push_back(pair<int, int>(IP1[j].first, IP1[j].second));
				}
			}
			else if (cnt_2 == 6) {
				for (int i = 0; i < 2; i++) {
					IP1_Rate.push_back(pair<int, int>(IP1[i].first, IP1[i].second));
				}
				for (int j = 2; j < IP1.size(); j++) {
					IP1_Volume.push_back(pair<int, int>(IP1[j].first, IP1[j].second));
				}
			}
			else {
				for (int i = 0; i < cnt_2 + 1; i++) {
					IP1_Rate.push_back(pair<int, int>(IP1[i].first, IP1[i].second));
				}
				for (int j = cnt_2 + 1; j < IP1.size(); j++) {
					IP1_Volume.push_back(pair<int, int>(IP1[j].first, IP1[j].second));
				}
			}
			IP1.clear();

			//X좌표값을 기준으로 RATE 영역과 VOLUME 영역으로 구분하기 (IP2)			

			for (int i = 0; i < num_roi_IP2.size(); i++) {
				IP2.push_back(pair<int, int>(num_roi_IP2[i][0], num_roi_IP2[i][5]));
			}

			sort(IP2.begin(), IP2.end());
			num_roi_IP2.clear();

			//X좌표값을 기준으로 RATE 영역과 VOLUME 영역으로 구분하기 (IP2)
			int cnt_3 = 0;
			for (int i = 0; i < IP2.size() - 1; i++) {

				if (abs(IP2[i].first - IP2[i + 1].first) > 3 * width_avg) {
					break;
				}

				cnt_3++;

			}

			// VOLUME 수치의 자릿수에 따라 RATE와 VOLUME 영역으로 세밀하게 구분하기 (IP2)

			if (cnt_3 == 8) {
				for (int i = 0; i < 4; i++) {
					IP2_Rate.push_back(pair<int, int>(IP2[i].first, IP2[i].second));
				}
				for (int j = 4; j < IP2.size(); j++) {
					IP2_Volume.push_back(pair<int, int>(IP2[j].first, IP2[j].second));
				}
			}
			else if (cnt_3 == 7) {
				for (int i = 0; i < 3; i++) {
					IP2_Rate.push_back(pair<int, int>(IP2[i].first, IP2[i].second));
				}
				for (int j = 3; j < IP2.size(); j++) {
					IP2_Volume.push_back(pair<int, int>(IP2[j].first, IP2[j].second));
				}
			}
			else if (cnt_3 == 6) {
				for (int i = 0; i < 2; i++) {
					IP2_Rate.push_back(pair<int, int>(IP2[i].first, IP2[i].second));
				}
				for (int j = 2; j < IP2.size(); j++) {
					IP2_Volume.push_back(pair<int, int>(IP2[j].first, IP2[j].second));
				}
			}
			else {
				for (int i = 0; i < cnt_3 + 1; i++) {
					IP2_Rate.push_back(pair<int, int>(IP2[i].first, IP2[i].second));
				}
				for (int j = cnt_3 + 1; j < IP2.size(); j++) {
					IP2_Volume.push_back(pair<int, int>(IP2[j].first, IP2[j].second));
				}
			}
			IP2.clear();

			// IP1 와 IP2의 최종 수치를 계산하는 부분
			float IP1_rate;

			if (IP1_Rate.size() == 4) {
				IP1_rate = 100 * IP1_Rate[0].second + 10 * IP1_Rate[1].second + 1 * IP1_Rate[2].second + 0.1 * IP1_Rate[3].second;
			}
			else if (IP1_Rate.size() == 3) {
				IP1_rate = 10 * IP1_Rate[0].second + 1 * IP1_Rate[1].second + 0.1 * IP1_Rate[2].second;
			}
			else if (IP1_Rate.size() == 2) {
				IP1_rate = 1 * IP1_Rate[0].second + 0.1 * IP1_Rate[1].second;
			}

			float IP1_volume;

			if (IP1_Volume.size() == 5) {
				IP1_volume = 1000 * IP1_Volume[0].second + 100 * IP1_Volume[1].second + 10 * IP1_Volume[2].second + 1 * IP1_Volume[3].second + 0.1 * IP1_Volume[4].second;
			}
			else if (IP1_Volume.size() == 4) {
				IP1_volume = 100 * IP1_Volume[0].second + 10 * IP1_Volume[1].second + 1 * IP1_Volume[2].second + 0.1 * IP1_Volume[3].second;
			}
			else if (IP1_Volume.size() == 3) {
				IP1_volume = 10 * IP1_Volume[0].second + 1 * IP1_Volume[1].second + 0.1 * IP1_Volume[2].second;
			}
			else if (IP1_Volume.size() == 2) {
				IP1_volume = 1 * IP1_Volume[0].second + 0.1 * IP1_Volume[1].second;
			}

			float IP2_rate;

			if (IP2_Rate.size() == 4) {
				IP2_rate = 100 * IP2_Rate[0].second + 10 * IP2_Rate[1].second + 1 * IP2_Rate[2].second + 0.1 * IP2_Rate[3].second;
			}
			else if (IP2_Rate.size() == 3) {
				IP2_rate = 10 * IP2_Rate[0].second + 1 * IP2_Rate[1].second + 0.1 * IP2_Rate[2].second;
			}
			else if (IP2_Rate.size() == 2) {
				IP2_rate = 1 * IP2_Rate[0].second + 0.1 * IP2_Rate[1].second;
			}
			
			float IP2_volume;

			if (IP2_Volume.size() == 5) {
				IP2_volume = 1000 * IP2_Volume[0].second + 100 * IP2_Volume[1].second + 10 * IP2_Volume[2].second + 1 * IP2_Volume[3].second + 0.1 * IP2_Volume[4].second;
			}
			else if (IP2_Volume.size() == 4) {
				IP2_volume = 100 * IP2_Volume[0].second + 10 * IP2_Volume[1].second + 1 * IP2_Volume[2].second + 0.1 * IP2_Volume[3].second;
			}
			else if (IP2_Volume.size() == 3) {
				IP2_volume = 10 * IP2_Volume[0].second + 1 * IP2_Volume[1].second + 0.1 * IP2_Volume[2].second;
			}
			else if (IP2_Volume.size() == 2) {
				IP2_volume = 1 * IP2_Volume[0].second + 0.1 * IP2_Volume[1].second;
			}


			if (!IP1_Rate.empty() & !IP1_Volume.empty()) {
				cout << "IP1_RATE(ML/H) :" << " " << IP1_rate << endl;
				cout << "IP1_VOLUM(ML) :" << " " << IP1_volume << endl;
			}	

			if (!IP2_Rate.empty() & !IP2_Volume.empty()) {
				cout << "IP2_RATE(ML/H) :" << " " << IP2_rate << endl;
				cout << "IP2_VOLUM(ML) :" << " " << IP2_volume << endl;	
			}			

		}
		
		IP1_Rate.clear();
		IP1_Volume.clear();					
		IP2_Rate.clear();
		IP2_Volume.clear();		
		
		
		imshow("frame", frame);
		imshow("V_plane", V_plane);
		imshow("frame_new", frame_new);
		imshow("frame_raw", frame_raw);

		
		if (waitKey(100) == 27)
			break;
			
	}

	return 0;

}