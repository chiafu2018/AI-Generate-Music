# AI_generate_music  
I use 3 different kind of neural network to generate piano music and compare their differences. Moreover, you can also download youtube playlists as your dataset. However, the outcomes aren't very good becuase the audio_to_midi can't convert the mp3 to mid well.  
  
My environment:  
Windows 11  
Create environments in Anaconda  
Python 3.8.16  
CudaToolkit 11.2  
cudnn=8.1.0  
fluidsynth 2.2.4  

You can check this websites for details: https://reurl.cc/Eopo2k
  
If you want to download youtube music online, you should pay attention for the following details !!  
Some libraries you need to pay attention:  
pip install youtube2midi  
pip uninstall youtube-dl    #The latest version can't be used !!  
pip install youtube-dl==2020.12.9   #Please download this version !!  
