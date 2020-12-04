from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves
import tarfile, os
import random
import json
import urllib
from GnuGo import *

def rev_coord_to_name(coord):
    if coord == (-1, -1):
        return 'PASS'
    letterIndex = "ABCDEFGHJ"
    return letterIndex[coord[1]]+str(coord[0]+1)


def download_pro_games():
    raw_samples_file = "9x9_2020_09.tar.bz2"

    if not os.path.isfile(raw_samples_file):
        print("File", raw_samples_file, "not found, I am downloading it...", end="")
        urllib.request.urlretrieve("http://www.yss-aya.com/cgos/9x9/archives/9x9_2020_09.tar.bz2", raw_samples_file)
        print(" Done")
    
    os.makedirs('pro_data', exist_ok=True)
    tar = tarfile.open("9x9_2020_09.tar.bz2", "r:bz2")  
    tar.extractall('./pro_data')
    tar.close()

def get_file_names(download=True):

    if download:
        download_pro_games()

    return(getListOfFiles('./pro_data'))

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def monte_carlo(gnugo, moves, nbsamples = 10):

    list_of_moves = [] # moves to backtrack
    #if len(moves) > 1 and moves[-1] == "PASS" and moves[-2] == "PASS":
    #    return None # FIXME
    epochs = 0 # Number of played games
    toret = []
    black_wins = 0
    white_wins = 0
    black_points = 0
    white_points = 0
    while epochs < nbsamples:
        print(epochs)
        while True:
            m = moves.get_randomized_best()
            # print(epochs, m)
            r = moves.playthis(m)
            if r == "RES": 
                return None
            list_of_moves.append(m)
            if len(list_of_moves) > 1 and list_of_moves[-1] == "PASS" and list_of_moves[-2] == "PASS":
                break
        score = gnugo.finalScore()
        toret.append(score)
        if score[0] == "W":
            white_wins += 1
            if score[1] == "+":
                white_points += float(score[2:])
        elif score[0] == "B":
            black_wins += 1
            if score[1] == "+":
                black_points += float(score[2:])
        (ok, res) = gnugo.query("gg-undo " + str(len(list_of_moves)))
        list_of_moves = []
        epochs += 1
    return epochs, toret, black_wins, white_wins, black_points, white_points


def get_pro_tables():
    out = []

    all_filenames = get_file_names()
    
    count_prob = 0
    count_ok = 0

    for filename in all_filenames:

        f = open(filename, "rb")
        sgf_src = f.read()
        f.close()

        try:
            sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except Exception:
            continue
        
        if len(plays) < 5:
            continue
        
        end_idx = random.randint(5, len(plays))

        new_game = {'list_of_moves': [], 'black_stones': [], 'white_stones': []}
        
        skip = False
        turn = 0
        for play in plays[:end_idx]:
            
            if play[1] == None:
                skip = True
                continue
            else:
                new_game['list_of_moves'].append(rev_coord_to_name(play[1]))
                
                if turn == 0:
                    new_game['black_stones'].append(rev_coord_to_name(play[1]))
                    turn = 1
                elif turn == 1:
                    new_game['white_stones'].append(rev_coord_to_name(play[1]))
                    turn = 0
                
        if not skip:
            assert len(new_game['black_stones']) + len(new_game['white_stones']) == len(new_game['list_of_moves'])
            new_game['depth'] = len(new_game['list_of_moves'])    
            out.append(new_game)
            
    return out

print('SCRIPT IS RUNNING')
gnugo = GnuGo(9)
out = []
tables = get_pro_tables()

print(len(tables))

print('will start')
for idx, table in enumerate(tables):
    print('game ' + str(idx))
    print(table)
    moves = gnugo.Moves(gnugo)
    depth = 0

    for move in table['list_of_moves']:
        retvalue = moves.playthis(move)
        if retvalue == "ERR":
            print("ERR")
            break
            
    list_of_moves = table['list_of_moves']
    blackstones = table['black_stones']
    whitestones = table['white_stones']
    
    (epochs, scores, black_wins, white_wins, black_points, white_points) = monte_carlo(gnugo, moves, 100)
    summary = {"depth": len(list_of_moves), "list_of_moves": list_of_moves, 
            "black_stones":blackstones, "white_stones": whitestones,
            "rollouts":epochs, 
            #"detailed_results": scores, 
            "black_wins":black_wins, "black_points": black_points,
            "white_wins":white_wins, "white_points":white_points}
    
    out.append(summary)

    print(out)

    gnugo.query("clear_board")

with open('pro_games.json', 'w') as fp:
    json.dump(out, fp)