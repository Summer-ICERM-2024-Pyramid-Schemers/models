import matplotlib.pyplot as plt
import os.path

DEFAULT_SEASONS = range(2010,2024)
SKIP_FIRST_2_SEASONS = range(DEFAULT_SEASONS.start+2,DEFAULT_SEASONS.stop)
# This list should be the names of the leagues as they should appear in plotting
# This only works properly when the ids begin at 1 and have a consistent step of 1
ALL_LEAGUES = ["Premier League","Championship","League One","League Two","Bundesliga","2. Bundesliga","Scottish Premiership","Scottish Championship", "Scottish League One", "Scottish League Two"]
COUNTRY_TO_LEAGUES = {None:[1,2,3,4,5,6,7,8,9,10], "all":[1,2,3,4,5,6,7,8,9,10], "england":[1,2,3,4], "germany":[5,6], "scotland":[7,8,9,10]}
COUNTRY_TO_ADJECTIVES = {"england":"English", "germany":"German", "scotland":"Scottish"}

IMAGES_FILEPATH = os.path.abspath(os.path.join(__file__,"../../images"))
if not os.path.exists(IMAGES_FILEPATH):
    os.mkdir(IMAGES_FILEPATH)

def savefig_to_images_dir(filename):
    plt.savefig(os.path.join(IMAGES_FILEPATH,filename))
