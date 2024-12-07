#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <fstream>
#include <windows.h>
#include <iostream>
#include <cstdlib>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#include <regex>

std::ofstream debugLog("debug_log.txt", std::ios::out); // ����� �α� ����


void logMessage(const std::string& message) {
    debugLog << message << std::endl; // ���Ͽ� �޽��� ����
    std::cout << message << std::endl; // �ֿܼ��� ���
}

using namespace cv;
using namespace std;

void imageProcessing(string input);     // Image processing to extract strings form car plates  
void printCarNumber();                  // Print car number
string getCarNumber(string text);       // Use regular expression to get car number
char* UTF8ToANSI(const char* pszCode);  // To prevent korean language broken

int main()
{
    imageProcessing("carImage/4.jpg");
    printCarNumber();

    //for (int i = 1; i < 11; i++) {    // All images in carImage will be processed
    //    char buf[256];
    //    sprintf_s(buf, "carImage/%d.jpg", i);
    //    imageProcessing(buf);
    //    printCarNumber();

    //    Mat image = imread("carImage/temp.jpg");
    //    sprintf_s(buf, "carImage/%d-1.jpg", i);
    //    imwrite(buf, image);
    //}

    return 0;
}

void imageProcessing(string input) {
    // �̹��� �ε�
    Mat image = imread(input);
    if (image.empty()) {
        cerr << "Error: Could not load image: " << input << endl;
        return;
    }
    imshow("Original Image", image);
    waitKey(0); // ����ڰ� Ű �Է��� ������ ���

    // ���� �ʱ�ȭ
    Mat grayImage, blurredImage, edgeImage, drawing;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // 1. �׷��̽����� ��ȯ
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    imshow("Gray Image", grayImage);
    waitKey(0);

    // 2. ���� (������ ����)
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);
    imshow("Blurred Image", blurredImage);
    waitKey(0);

    // 3. ���� ���� (Canny)
    Canny(blurredImage, edgeImage, 100, 300, 3);
    imshow("Edge Detection", edgeImage);
    waitKey(0);

    // 4. ������ ���� (findContours)
    findContours(edgeImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    cout << "Detected contours: " << contours.size() << endl;

    // ������ �ٻ�ȭ �� �ٿ�� �ڽ� ����
    vector<vector<Point>> contoursPoly(contours.size());
    vector<Rect> boundRects(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], contoursPoly[i], 3, true);
        boundRects[i] = boundingRect(contoursPoly[i]);
    }

    // 5. ���͸� (��ȣ�� �ĺ� ���� ����)
    drawing = Mat::zeros(edgeImage.size(), CV_8UC3);
    vector<Rect> filteredRects;
    for (size_t i = 0; i < contours.size(); i++) {
        double ratio = (double)boundRects[i].height / boundRects[i].width;
        double area = boundRects[i].area();
        // ��ȣ�� Ư���� �´� ���͸� ����
        if (ratio >= 0.5 && ratio <= 2.5 && area >= 100 && area <= 700) {
            rectangle(drawing, boundRects[i].tl(), boundRects[i].br(), Scalar(0, 255, 0), 2);
            filteredRects.push_back(boundRects[i]);
        }
    }

    imshow("Filtered Rectangles", drawing);
    waitKey(0);

    // 6. ��ȣ�� �ĺ� ���� (���� �� ����)
    sort(filteredRects.begin(), filteredRects.end(),
        [](const Rect& a, const Rect& b) { return a.tl().x < b.tl().x; });

    // 7. ��ȣ�� �̹��� ����
    if (!filteredRects.empty()) {
        // �����: Filtered Rectangles Ȯ��
        logMessage("Filtered Rectangles: " + std::to_string(filteredRects.size()));
        for (const auto& rect : filteredRects) {
            logMessage("Filtered Rect: [x=" + std::to_string(rect.x) +
                ", y=" + std::to_string(rect.y) +
                ", w=" + std::to_string(rect.width) +
                ", h=" + std::to_string(rect.height) + "]");
        }

        try {
            // 1. ���յ� �ĺ� ���� (X ��ǥ ����)
            sort(filteredRects.begin(), filteredRects.end(),
                [](const Rect& a, const Rect& b) { return a.tl().x < b.tl().x; });

            // 2. �ĺ� �׷�ȭ (Y ��ǥ �� X ���� ����)
            vector<vector<Rect>> groupedRects;
            double yThreshold = 50; // ���� �ٷ� �ν��� Y ��ǥ ���� �Ӱ谪
            double xThreshold = 50; // ���� �׷����� ���� X ��ǥ ���� �Ӱ谪

            for (const auto& rect : filteredRects) {
                bool grouped = false;
                for (auto& group : groupedRects) {
                    // ���� �׷����� ���� ����: Y ��ǥ�� X ���� ���
                    if (abs(group[0].tl().y - rect.tl().y) < yThreshold &&
                        abs(group.back().tl().x - rect.tl().x) < xThreshold) {
                        group.push_back(rect);
                        grouped = true;
                        break;
                    }
                }
                if (!grouped) {
                    groupedRects.push_back({ rect });
                }
            }


            // �����: �׷�ȭ�� �ĺ� Ȯ��
            logMessage("Grouped Rectangles: " + std::to_string(groupedRects.size()));
            for (size_t i = 0; i < groupedRects.size(); ++i) {
                logMessage("Group " + std::to_string(i + 1) + " - Size: " + std::to_string(groupedRects[i].size()));
                for (const auto& rect : groupedRects[i]) {
                    logMessage("Rect: [x=" + std::to_string(rect.x) +
                        ", y=" + std::to_string(rect.y) +
                        ", w=" + std::to_string(rect.width) +
                        ", h=" + std::to_string(rect.height) + "]");
                }
            }


            // 3. �׷�ȭ�� �ĺ����� �簢�� ������ ���͸�
            vector<vector<Rect>> validGroups;
            for (const auto& group : groupedRects) {
                if (group.size() == 7 || group.size() == 8) { // ��ȣ�� Ư���� 7�� �Ǵ� 8��
                    validGroups.push_back(group);
                }
                else if (group.size() > 3) { // 3�� �̻��� ��� �߰� ����
                    bool isPotentialPlate = true;
                    for (size_t i = 1; i < group.size(); ++i) {
                        // �簢�� ũ�� �� ���� ����
                        double widthDiff = abs(group[i].width - group[i - 1].width);
                        double heightDiff = abs(group[i].height - group[i - 1].height);
                        double xGap = abs(group[i].tl().x - group[i - 1].br().x);

                        if (widthDiff > 15 || heightDiff > 15 || xGap > 100) {
                            isPotentialPlate = false;
                            break;
                        }
                    }
                    if (isPotentialPlate) validGroups.push_back(group);
                }
            }


            // �����: ��ȿ�� �׷� Ȯ��
            logMessage("Valid Groups: " + std::to_string(validGroups.size()));
            for (const auto& group : validGroups) {
                for (const auto& rect : group) {
                    logMessage("Valid Rect: [x=" + std::to_string(rect.x) +
                        ", y=" + std::to_string(rect.y) +
                        ", w=" + std::to_string(rect.width) +
                        ", h=" + std::to_string(rect.height) + "]");
                }
            }

            // 4. �迭 ��� �߰� ����
            vector<Rect> finalCandidates;
            for (const auto& group : validGroups) {
                bool isHorizontal = true;
                double avgY = 0.0;
                for (const auto& rect : group) avgY += rect.tl().y;
                avgY /= group.size();

                for (const auto& rect : group) {
                    if (abs(rect.tl().y - avgY) > 20) { // Y ��ǥ�� �ʹ� �ٸ��� ���� �迭 �ƴ�
                        isHorizontal = false;
                        break;
                    }
                }

                if (isHorizontal || group.size() == 8) { // ���� �迭�̰ų� ���� �迭(���� ������ ��ȣ��)
                    for (const auto& rect : group) {
                        finalCandidates.push_back(rect);
                    }
                }
            }

            // �����: ���� �ĺ� Ȯ��
            logMessage("Final Candidates: " + std::to_string(finalCandidates.size()));
            if (!finalCandidates.empty()) {
                // �����: Final Candidates Ȯ��
                logMessage("Final Candidates (before boundingRect): " + std::to_string(finalCandidates.size()));
                for (const auto& rect : finalCandidates) {
                    logMessage("Rect: [x=" + std::to_string(rect.x) +
                        ", y=" + std::to_string(rect.y) +
                        ", w=" + std::to_string(rect.width) +
                        ", h=" + std::to_string(rect.height) + "]");
                }

                // ���� ó��: finalCandidates�� ��� �ִ� ���
                if (finalCandidates.empty()) {
                    logMessage("Error: finalCandidates is empty. Cannot compute boundingRect.");
                    return;
                }

                // boundingRect�� ����� Point ���� ����
                vector<Point> candidatePoints;
                for (const auto& rect : finalCandidates) {
                    candidatePoints.push_back(rect.tl()); // ���� ��� ��ǥ �߰�
                    candidatePoints.push_back(rect.br()); // ������ �ϴ� ��ǥ �߰�
                }

                // boundingRect ȣ��
                Rect plateRect = boundingRect(candidatePoints); // �ĺ��� ���δ� ���� Rect
                logMessage("Bounding Rect computed: [x=" + std::to_string(plateRect.x) +
                    ", y=" + std::to_string(plateRect.y) +
                    ", w=" + std::to_string(plateRect.width) +
                    ", h=" + std::to_string(plateRect.height) + "]");

                // �ĺ� ���� Ȯ��
                plateRect.x = max(plateRect.x - 20, 0);
                plateRect.y = max(plateRect.y - 10, 0);
                plateRect.width = min(plateRect.width + 40, image.cols - plateRect.x);
                plateRect.height = min(plateRect.height + 20, image.rows - plateRect.y);

                Mat plateImage = image(plateRect);
                imshow("Final Plate Candidate", plateImage);
                waitKey(0);
                // OCR ��ó��
                cvtColor(plateImage, plateImage, COLOR_BGR2GRAY);
                resize(plateImage, plateImage, Size(plateImage.cols * 3, plateImage.rows * 3), 0, 0, INTER_LINEAR); // 3�� Ȯ��
                adaptiveThreshold(plateImage, plateImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 10);
                imwrite("carImage/temp.jpg", plateImage);
                imshow("Processed Plate", plateImage);
                waitKey(0);

                // OCR ��� ���
                tesseract::TessBaseAPI ocr;
                if (ocr.Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "kor", tesseract::OEM_LSTM_ONLY)) {
                    logMessage("Tesseract initialization failed!");
                    return;
                }
                ocr.SetImage(plateImage.data, plateImage.cols, plateImage.rows, 1, plateImage.step);
                string ocrResult = ocr.GetUTF8Text();
                logMessage("Detected Plate Text: " + ocrResult);
                ocr.End();
            }
            else {
                logMessage("No valid plate candidate found!");
            }
        }
        catch (const std::exception& e) {
            logMessage("Exception caught: " + std::string(e.what()));
        }
    }
    else {
        logMessage("No suitable plate candidate found!");
    }
}


void printCarNumber() {
    Mat image;
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

    // Initialize tesseract-ocr with Korean, oem is 0
    if (api->Init("C:\\Program Files\\tesseract-OCR\\tessdata", "kor3", tesseract::OEM_TESSERACT_ONLY)) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Set page segmentation mode to PSM_SINGLE_LINE(7), it assumes there is only one line
    api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);

    // Open car plate image
    image = imread("carImage/temp.JPG");
    resize(image, image, cv::Size(image.cols * 2, image.rows * 2), 0, 0, 1);
    imshow("EnlargedCarPlate", image);
    waitKey(0);

    // Open input image with leptonica library
    Pix* carNumber = pixRead("carImage/temp.jpg");
    api->SetImage(carNumber);
    // Get OCR result
    string outText = api->GetUTF8Text();
    string text = UTF8ToANSI(outText.c_str());
    text = getCarNumber(text);
    cout << "carNumber : " << text << endl;

    // Destroy used object and release memory
    api->End();
    delete api;
    pixDestroy(&carNumber);
}

string getCarNumber(string text) {
    int i = 0;

    //cout << test << '\n';

    // Extract "12��3456" or "123��4567"
    regex re("\\d{2,3}\\W{2}\\s{0,}\\d{4}");
    smatch match;
    if (regex_search(text, match, re)) {
        return match.str();
    }
    else {
        return "0";
    }
}

char* UTF8ToANSI(const char* pszCode) // For korean
{
    BSTR    bstrWide;
    char* pszAnsi;
    int     nLength;

    // Get nLength of the Wide Char buffer
    nLength = MultiByteToWideChar(CP_UTF8, 0, pszCode, strlen(pszCode) + 1, NULL, NULL);
    bstrWide = SysAllocStringLen(NULL, nLength);

    // Change UTF-8 to Unicode (UTF-16)
    MultiByteToWideChar(CP_UTF8, 0, pszCode, strlen(pszCode) + 1, bstrWide, nLength);

    // Get nLength of the multi byte buffer
    nLength = WideCharToMultiByte(CP_ACP, 0, bstrWide, -1, NULL, 0, NULL, NULL);
    pszAnsi = new char[nLength];

    // Change from unicode to mult byte
    WideCharToMultiByte(CP_ACP, 0, bstrWide, -1, pszAnsi, nLength, NULL, NULL);
    SysFreeString(bstrWide);

    return pszAnsi;
}