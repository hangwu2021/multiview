#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
using namespace Eigen;

/* 相机参数 */
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

// 基线
double b = 0.573;

/**
 * DECLARE FUNCTION 
*/
void find_feature_matches(
    const cv::Mat& img_1, 
    const cv::Mat& img_2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::DMatch>& matches
);

void pose_estimate_2d2d(
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& K,
    cv::Mat& R, 
    cv::Mat& t
);

cv::Point3d pixel2Cam(const cv::Point2d& p, const cv::Mat& K);

/**
 *  MAIN FUNCTION 
*/
int main(int argc, char* argv[])
{
    cv::Mat image_left = cv::imread("images/left.png"), image_right = cv::imread("images/right.png");
    if (image_left.data == nullptr || image_right.data == nullptr)
    {
        std::cout << "Load Image Failed!" << std::endl;
        return 0;
    }

    // SGBM
    cv::Mat disparity_SGBM, disparity;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32);

    sgbm->compute(image_left, image_right, disparity_SGBM);
    std::cout << "disparity.size() = " << disparity_SGBM.size() << ", disparity.channels() = " << disparity_SGBM.channels() << std::endl;

    disparity_SGBM.convertTo(disparity, CV_32F, 1.0/16.0);

    // Point Cloud
    std::vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointCloud;
    for (int v = 0; v < disparity.rows; v++)
    {
        for (int u = 0; u < disparity.cols; u++)
        {
            // by pass Outliers
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0)
            {
                continue;
            }
            
            Vector4d point(0, 0, 0, image_left.at<uchar>(v, u)/255.0);
            double x = (u-cx)/fx;
            double y = (v-cy)/fy;
            double depth = fx * b / disparity.at<float>(v, u);

            point[0] = x;
            point[1] = y;
            point[2] = depth;

            pointCloud.push_back(point);
        }
    }

    std::cout << "pointCloud.size() = " << pointCloud.size() << std::endl;
    
    // max_element
    Vector4d maxDisparityPoint = Vector4d(0, 0, 0, 0);
    double meanDisparity = 0.0, sumDisparity = 0.0;
    for (int i = 0; i < pointCloud.size(); ++i)
    {
        if (maxDisparityPoint[2] < pointCloud[i][2])
        {
            maxDisparityPoint = pointCloud[i];
        }

        sumDisparity += pointCloud[i][2];
    }

    meanDisparity = sumDisparity / pointCloud.size();

    std::cout << maxDisparityPoint << std::endl;
    std::cout << "meanDisparity = " << meanDisparity << std::endl;

//    cv::imwrite("output/disparity_SGBM.png", disparity_SGBM);
//    cv::imwrite("output/disparity.png", disparity);
// -------------------------------------------------------------------------------------------------
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;

    find_feature_matches(image_left, image_right, keypoints_1, keypoints_2, matches);
    std::cout << "matches.size() = " << matches.size() << std::endl;

    cv::Mat result;
    cv::drawMatches(image_left, keypoints_1, image_right, keypoints_2, matches, result);
    cv::imwrite("output/keypoints_left.png", result);

    // 相机内参,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // estimate transform matrix 
    cv::Mat R, t;
    pose_estimate_2d2d(keypoints_1, keypoints_2, matches, K, R, t);

    std::cout << "R = \n" << R << "\nt = \n" << t << std::endl;

    // t^
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0), 
                                            t.at<double>(2, 0), 0, -t.at<double>(0, 0), 
                                            -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    // Validate
    std::vector<double> errors;
    for (const cv::DMatch& m : matches)
    {
        cv::Point3d pt1 = pixel2Cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, pt1.z);
        
        cv::Point3d pt2 = pixel2Cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, pt2.z);

        double e = cv::Mat_<double>(y2.t() * t_x * R * y1)(0, 0);
        errors.push_back(e);
    }

    // Mean and Std
    double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    double mean = sum / errors.size();
    double stdErr = 0.;

    double temp = 0.;
    std::for_each(errors.begin(), errors.end(), [&](const double d){
        temp += std::pow(d - mean, 2);
    });

    stdErr = std::sqrt(temp / (errors.size() - 1));

    std::cout << " epipolar constraint = " << mean << ", " << stdErr << std::endl;

    return 0;
}

/**
 * DEFINE FUNCTION 
*/
void find_feature_matches(
    const cv::Mat& img_1, 
    const cv::Mat& img_2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::DMatch>& matches
)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // Oriented FAST 
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // BRIEF 
    cv::Mat descriptors_1, descriptors_2;
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // Match 
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // Distance 
    double min_dist = 10000, max_dist = 0, dist = 0;
    for (int i = 0; i < descriptors_1.rows; ++i)
    {
        dist = match[i].distance;

        if (dist < min_dist)
        {
            min_dist = dist;
        }

        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }

    std::cout << "min_dist = " << min_dist << "\n";
    std::cout << "max_dist = " << max_dist << "\n";

    // Optical KeyPoints
    for (int i = 0; i < descriptors_1.rows; ++i)
    {
        if (match[i].distance < std::min(min_dist * 20, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimate_2d2d(
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& K,
    cv::Mat& R, 
    cv::Mat& t
)
{
    // KeyPoint -> Point2f
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < (int)matches.size(); ++i)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // Essential Matrix 
    cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
    double focal_length = (K.at<double>(0, 0) + K.at<double>(1, 1)) / 2;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);

    // recover R and t from Essential Matrix 
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

cv::Point3d pixel2Cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point3d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1), 
        1.0
    );
}
