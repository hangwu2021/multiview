#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

void orb_features(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

void match_min(const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& good_matches);

void ransac(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& queryKeyPoint, const std::vector<cv::KeyPoint>& trainKeyPoint, std::vector<cv::DMatch>& ransac_matches);


int main(int argc, char* argv[])
{
    cv::Mat img1 = cv::imread("/home/ai2/workspace/hang.wu/multiview/ch7/input/1.png", cv::IMREAD_GRAYSCALE),
            img2 = cv::imread("/home/ai2/workspace/hang.wu/multiview/ch7/input/2.png", cv::IMREAD_GRAYSCALE);
    if (img1.data == nullptr || img2.data == nullptr)
    {
        std::cerr << "Image Load Failed!" << std::endl;
        return -1;
    }

    // Orb detect 
    std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    cv::Mat descriptors1, descriptors2;
    orb_features(img1, keyPoints1, descriptors1);
    orb_features(img2, keyPoints2, descriptors2);
    
    // Orb match 
    std::vector<cv::DMatch> matches, good_matches, ransac_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);
    std::cout << "matches.size = " << matches.size() << std::endl;

    // good_matches 
    match_min(matches, good_matches);
    std::cout << "good_matches.size = " << good_matches.size() << std::endl;

    // find homography
    ransac(good_matches, keyPoints1, keyPoints2, ransac_matches);
    std::cout << "ransac_matches.size() = " << ransac_matches.size() << std::endl;
    

    return 0;
}


void orb_features(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    orb->detect(gray, keypoints);
    orb->compute(gray, keypoints, descriptors);
}

void match_min(const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& good_matches)
{
    double min_dist = 10000.0, max_dist = 0.;
    for (int i = 0; i < (int)matches.size(); ++i)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
        {
            min_dist = dist;
        }
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }

    std::cout << "min_dist = " << min_dist << std::endl;
    std::cout << "max_dist = " << max_dist << std::endl;

    for (int i = 0; i < (int)matches.size(); ++i)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 20.))
        {
            good_matches.push_back(matches[i]);
        }
    }
}

void ransac(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& queryKeyPoint, const std::vector<cv::KeyPoint>& trainKeyPoint, std::vector<cv::DMatch>& ransac_matches)
{
    std::vector<cv::Point2f> queryPoints(matches.size()), trainPoints(matches.size());
    for (int i = 0; i < (int)matches.size(); ++i)
    {
        queryPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
        trainPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
    }

    std::vector<int> inliersMask(queryPoints.size());
    cv::Mat h = cv::findHomography(queryPoints, trainPoints, cv::RANSAC, 5, inliersMask);
    std::cout << "homography = \n" << h << std::endl;
    for (int i = 0; i < inliersMask.size(); ++i)
    {
        if (inliersMask[i])
        {
            ransac_matches.push_back(matches[i]);
        }
    }
}
