import os
from music21 import *
from pytube import *
from youtube2midi import download_and_convert
import shutil
from tqdm import tqdm

'''''''''''''''''''''''''''''''''''''''''
From Youtube download mp3 and convert to midi files
Cmd: 
pip install youtube2midi
pip uninstall youtube-dl    #The latest version can't be used !!
pip install youtube-dl==2020.12.9   #Please download this version !!
'''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''
Check if one video is okay
yt = YouTube('https://www.youtube.com/watch?v=PUQlJAlWGb8')
download_and_convert('https://www.youtube.com/watch?v=PUQlJAlWGb8',output_name=f'{yt.title}.mid')
shutil.move(f'{yt.title}.mid',"C:\\Users\chiaf\\source_vs\\AI_project_new\\Source Data\\Download Online")
'''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''
Useful Data Resource Playlist

抖音歌曲: https://www.youtube.com/watch?v=d9WsJ-mzbho&list=PLATKOC0vXmk4oSNlJ5Slliyo3gp-paPIR
Hip hop: https://www.youtube.com/watch?v=R4nAvDXpvHk&list=RDQMn9PbdWPfMmI&start_radio=1
Boom Bap: https://www.youtube.com/watch?v=y5dXH3__eos&list=RDQM54OI6IAvgZU&index=2
Trap: https://www.youtube.com/watch?v=SKBsdOC8LJU&list=RDQMs5BWZb31JtY&start_radio=1
Drill Beat: https://www.youtube.com/watch?v=xjP5BUkT7ts&list=PLx3a5xbWJQGPBow1bAJSwO3qbwat-lAG0

'''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''
Default Composer list
'''''''''''''''''''''''
composer_list=[
    ('Albeniz',1),('Beethoven',2),('Borodin',3),('Brahms',4),('Burgmueller',5),
    ('Chopin',6),('Debussy',7),('Grieg',8),('Haydn',9),
    ('Liszt',10),('Mendelssohn',11),('Mozart',12),('Mussorgsky',13),
    ('Schubert',14),('Schumann',15),('Tchaikovsky',16),('Hip Hop Beats',17)
]

#Getting midi files
def capture_data(option):
    count=0
    if(option==1):
        url_playlist = input("Enter your url (don't put your url in ''): ")
        yt_list = Playlist(url_playlist)
        for url in yt_list.video_urls:  
            count+=1
            if(count<=50):
                try:
                    yt = YouTube(url)
                    print(f'Downloading: {yt.title}')
                    download_and_convert(url, output_name=str(count)+'.mid')
                    shutil.move(str(count)+'.mid','Source Data\\Download Online')
                except:
                    count-=1
                    print("Can't Download :(", end='\n')
        print("Finished!", end='\n')
        filepath = "Source Data\Download Online/"
    else:
        for i in composer_list:
            print("(",i[1],")",i[0])
        composer_num=int(input("Please enter the number of composer: "))
        composer_num-=1
        if composer_num < len(composer_list):
            selected_composer = composer_list[composer_num]
            print("The composer you want is ",selected_composer[0])
            filepath = "Source Data/" + selected_composer[0] + '/'
        else:
            print("Invalid!")

    #Loading midi files as stream
    all_midis= []
    count=0
    for file in os.listdir(filepath):
        if file.endswith(".mid"):
            count+=1
    print("Total number of dataset: ", count)
    progress = tqdm(total=count)
    for file in os.listdir(filepath):
        if file.endswith(".mid"):
            #print(file)    #Use for checking which data are unavailable
            tr = filepath+file
            midi = converter.parse(tr)  
            all_midis.append(midi)
            progress.update(1)

    return all_midis
      
def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


def chords_n_notes(Snippet):
    Melody = []
    offset = 0 #Incremental
    for i in Snippet:
        #If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") #Seperating the notes in chord
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)            
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        # pattern is a note
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)   
    return Melody_midi


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Check which data are unavailable and delete them "MANUALLY"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#all_midis= capture_data(0)
