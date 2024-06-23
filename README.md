# Posture Correction
> [!CAUTION]
> This is a very early WIP/prototype

The idea is simple: Use the webcam and some magic to detect bad posture and alert with a sound.


## Installation
```sh
pip install -r requirements.txt
```

## Running
```sh
python posture_correction.py
```

## Debugging
In debug mode, the webcam input is rendered with some landmarks and other information.
```sh
python posture_correction.py -d
# or
python posture_correction.py --debug
```



## Achknowledgements
- [This Article](https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/) was a great starting point.
- [The accompanying source code](https://github.com/spmallick/learnopencv/tree/master/Posture-analysis-system-using-MediaPipe-Pose) of the article.
- Sound effect from https://mixkit.co/free-sound-effects/ (specifically mixkit-alarm-tone-996.wav).

