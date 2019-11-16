Welcome to (my attempt at) the comma.ai Programming Challenge!
======

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Deliverable
-----

Your deliverable is test.txt. E-mail it to givemeajob@comma.ai, or if you think you did particularly well, e-mail it to George.

Evaluation
-----

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

Twitter
------

<a href="https://twitter.com/comma_ai">Follow us!</a>



Current Progress
-----------------


I have a pretty bad baseline which is trained on a really small segment of the data, due to memory limitations on my computer. I'd love to train this on a cluster, but I can't upload the entirety of the mp4 to github and transfer there since the video file is too large! What's worse is if you do  git clone of the original file, the mp4 file does not work properly. So for now I'll leave the basic baseline up along with the new way to format this file, and train it locally while adding a flash drive to my computer to host the data. 