# AI_generate_music  
I use 3 different kind of neural network to generate piano music and compare their differences. Moreover, you can also download youtube playlists as your dataset. However, the outcomes aren't very good becuase the audio_to_midi can't convert the mp3 to mid well.  

For songs in Output Data, I trained and generated them in my own computer.  
Welcome and happy to use these codes and dataset to generate music on your own !!  

My environment:  
Windows 11  
Create environments in Anaconda  
Python 3.8.16  
CudaToolkit 11.2  
cudnn=8.1.0  
fluidsynth 2.2.4  

You can check this websites for details: https://reurl.cc/Eopo2k
  
If you want to download youtube music online, you should pay attention to the following details!!  
Some libraries details you need to notice:  
pip install youtube2midi  
pip uninstall youtube-dl    #The latest version can't be used !!  
pip install youtube-dl==2020.12.9   #Please download this version !!  

For piano mid files, I download them in this website: http://www.piano-midi.de/  
However, there are some data can't be trained correctly, and I delete them manually.
Last, hope everybody can become a music producer !!!
