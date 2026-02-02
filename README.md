This is a music visualizer program I built to "express" music as a visible form of energy (colors, motion, shapes). I wanted to create cloud of energy that responds to the musical content and captures the "feeling" behind the music.

This music visualizer uses statistical analysis techniques and machine learning to process audio data. It learns the music as it plays and the visual elements will "dance" in a more structured way as it moves alongside the music.

Note: use Pyglet version 1.5.27 instead of latest Pyglet package

https://github.com/user-attachments/assets/f98c43af-6105-4f5a-baab-67d719c3c282

https://github.com/user-attachments/assets/dde5cadc-9500-4a15-8e33-ecb31f7ffc3e


***Statistical Analysis Techniques:***

Source Separation via Non-Negative Matrix Factorization:
https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf

Non-Negative Matrix Factorization - Multiplicative Update Method:
https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf

Deriving the Multiplicative Update Method:
https://stats.stackexchange.com/questions/351359/deriving-multiplicative-update-rules-for-nmf


***Machine Learning Techniques:***

Riemann Boltzmann Machine
https://github.com/echen/restricted-boltzmann-machines

Spiking Riemann Boltzmann Machine
https://github.com/Shikhargupta/Spiking-Restricted-Boltzmann-Machine


***Similarity Algorithms:***

Dynamic Time Warping
https://en.wikipedia.org/wiki/Dynamic_time_warping

https://databricks.com/blog/2019/04/30/understanding-dynamic-time-warping.html


***An example of the feature/temporal components produced by matrix factorization***

Results of using Non-Negative Matrix Factorization on an audio chunk:
Left side represents “feature” components. Right side represents "temporal" components.

<img width="691" height="535" alt="Screenshot 2026-02-01 223151" src="https://github.com/user-attachments/assets/b563e621-24bd-40b8-9de0-33ddb8d6db0b" />

<img width="611" height="451" alt="Screenshot 2026-02-01 223458" src="https://github.com/user-attachments/assets/f13b316b-a90c-4b5d-921b-14556b3b5609" />

<img width="615" height="455" alt="Screenshot 2026-02-01 223604" src="https://github.com/user-attachments/assets/5bab6b4b-c1ce-440e-a885-2f7695942c99" />

<img width="628" height="469" alt="Screenshot 2026-02-01 223613" src="https://github.com/user-attachments/assets/cfafa85d-0b26-4c69-ae17-53b5b965beb8" />

<img width="627" height="462" alt="Screenshot 2026-02-01 223650" src="https://github.com/user-attachments/assets/935a3f5f-a64b-4064-9c82-be7c843bd11b" />




