====================================================================
COMBINED EVALUATION — 4 VIDEO(S)
================================

CONTACT MODEL
Predicted-contact frames reviewed: 115
Actually contact:                  103
False positives:                   12
False positives among predictions: 10.4%
Precision on reviewed positives:   89.6%

KEYPOINT DETECTION
True-contact frames reviewed:      103
Keypoint detection succeeded:      103
Keypoint detection failed:         0
Detection success rate:            100.0%

Stance-leg split: left=48, right=55

STANCE-LEG KEYPOINT ERROR
Points compared:                   309
Mean pixel error:                  8.43 px
Mean normalized error:             3.62% of shank

## Keypoint          n     mean px     % shank

knee             103        7.43        3.06
ankle            103        6.18        2.88
heel             103       11.68        4.91

====================================================================
PER-VIDEO RESULTS
=================

Video 1
Contact precision:                 96.6%
False positives:                   1 / 29 (3.4%)
Keypoint detection success:        28 / 28 (100.0%)
Mean normalized keypoint error:    2.33% of shank

Video 2
Contact precision:                 85.2%
False positives:                   4 / 27 (14.8%)
Keypoint detection success:        23 / 23 (100.0%)
Mean normalized keypoint error:    2.77% of shank

Video 3
Contact precision:                 100.0%
False positives:                   0 / 30 (0.0%)
Keypoint detection success:        30 / 30 (100.0%)
Mean normalized keypoint error:    3.64% of shank

Video 4
Contact precision:                 75.9%
False positives:                   7 / 29 (24.1%)
Keypoint detection success:        22 / 22 (100.0%)
Mean normalized keypoint error:    6.12% of shank

====================================================================
SUMMARY
=======

The contact model achieved 89.6% precision across the four test videos.
Out of 115 frames predicted as contact, 103 were manually confirmed as
true contact and 12 were false positives.

The keypoint model successfully detected keypoints in all 103 manually
confirmed contact frames, giving a 100.0% detection success rate.

Across 309 manually evaluated stance-leg keypoints, the average
localization error was 8.43 pixels, corresponding to 3.62% of shank
length.

The ankle was the most accurately localized keypoint at 2.88% of shank
length, followed by the knee at 3.06%. The heel had the highest error
at 4.91%, indicating that heel localization is currently the weakest
part of the keypoint model.

Performance varied between videos. Contact precision ranged from 75.9%
to 100.0%, while normalized keypoint error ranged from 2.33% to 6.12%.
This suggests that recording conditions or differences between runners
can affect model performance.

Overall, the keypoint detector was highly reliable at producing
detections, while the main areas for improvement are reducing false
contact detections and improving heel localization.

NOTE:
The shank is the lower-leg segment between the knee and the ankle.
Using shank length to normalize the keypoint error makes the metric less
dependent on camera distance, image resolution, or the apparent size of
the runner in the frame. For example, an error of 3.62% of shank means
that the predicted keypoint was, on average, off by 3.62% of the
knee-to-ankle distance in the image.
