# Music-Genre-Classification
An independent project attempting to automate the classification of my digital music library by genre.


This repo contains the vitals of my music classification project which I completed in October 2019. 
This was intended as a demonstration to potential employers of the skills I had developed in independent study. It was also pretty fun.

Obviously, the music files themselves are only accessibe on my computer, but the code used to extract spectrometry data from the library is available in the Jupyter Notebook (using Python).
The rest, exploration, analysis and modeling, is contained in the following two R scripts.
"music_library.csv" contains the list of music files used in this project.
"Music_Classification.pdf" contains my final write-up.

--

Many months after completion, I realized that I should have found a way to normalize the amplitudes of the waveforms to control for the effects of the loudness war,
  so as to allow for better comparison between dynamic and brick-walled observations of the same genre.
For example, the last plot in the write-up compares a segment of the centroid features of two death metal tracks from the late 80's: **Pestilence**'s "Dehydrated" and **Death**'s "Leprosy".
Both of these tracks were recorded before the loudness war began, but one has a noticeably spikier centroid feature.
This is likely caused by "Dehydrated" being from the 2017 remaster of the album *Consuming Impulse*.
Thankfully, because of the mechanics of the graphical mapping method, this disparity does not have much of an effect on the prediction.
However, it would be good housekeeping to normalize the tracks.
Hopefully, I will return to this project at one point and do just that as well as update it with my current music collection.
