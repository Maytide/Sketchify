# https://raw.githubusercontent.com/asweigart/imgur-hosted-reddit-posted-downloader/master/imgur-hosted-reddit-posted-downloader.py

"""
Collect sample images from an anime-art themed subreddit (like /r/awwnime, SFW)
Saved as "image_id.ext", where url_hash is the url hash for each of the allowed hosts.
By default, downloads 100 images. Limit kwarg in collect_images() can be modified.
"""

import os, sys
import time
import datetime
import re
import glob

import praw
import requests
from bs4 import BeautifulSoup
from PIL import Image


IMAGE_FOLDER = 'images'
resize_dim = (256, 256)
RESIZE_FOLDER = 'resized_%s_%s' % resize_dim

MIN_SCORE = 100 # the default minimum score before it is downloaded
imgur_url_pattern = re.compile(r'(http://i.imgur.com/(.*))(\?.*)?')
target_subreddit = 'awwnime' # https://www.reddit.com/r/awwnime/
allowed_hosts = ('i.redd.it', 'cdn.awwni.me', 'i.imgur.com', 'imgur.com') # Only accept images from these hosts
check_types=('*.png', '*.jpg', '*.jpeg', '*.gif', '*.tiff', '*.bmp') # Only allow these kinds of images (preferably, only png + jpg)

CDN_AWWNI_ME = 'cdn_awwni_me'
REDDIT = 'ireddit'
IMGUR = 'imgur'

# Earliest submission acceptance threshold
start_date = datetime.date(2016, 1, 1)
start_unix = time.mktime(start_date.timetuple())

# Current time
end_unix = time.time()


def collect_anime_pics(limit=1000, daterange=None):

    def get_client():
        imgur_key_file = 'keys-reddit.txt'
        with open(imgur_key_file) as f:
            tokens = f.readlines()
            client_id = tokens[0].split(':')[1].strip()
            client_secret = tokens[1].split(':')[1].strip()
        reddit = praw.Reddit(client_id=client_id,
                             client_secret=client_secret,
                             user_agent='windows: collect anime images :LennyFace:')
        print(reddit.read_only)

        return reddit


    def get_flairs(r, start_unix=None, end_unix=None, sub='anime', verbose=False):
        r = get_client()

    def download_image(image_url, local_file_name, folder=IMAGE_FOLDER, id=0):
        savepath = os.path.join(folder, local_file_name)
        response = requests.get(image_url)
        if response.status_code == 200:
            print('Downloading #%d %s...' % (id, local_file_name))
            with open(savepath, 'wb') as fo:
                for chunk in response.iter_content(4096):
                    fo.write(chunk)

    def find_images(submissions=None, folder=IMAGE_FOLDER, limit=None):
		# Do not re-download existing files
        current_imgs = []
        for ftype in check_types:
            current_imgs.extend(glob.glob(os.path.join(folder, ftype)))
        print('Number of collected urls:', len(current_imgs))

        save_count = 0
        for i, submission in enumerate(submissions):
            print('-------------------------------------')
            if isinstance(limit, int) and save_count >= limit:
                print('Limit of %d reached! Returning' % limit)
                return

            print('%d: [%d / %s]' % ((i+1), save_count, str(limit)), 'Parsing:', submission.url)
            
            try:
                # Check for all the cases where we will skip a submission:
                if not any([host in submission.url for host in allowed_hosts]):
                    print('Nonaccepted host:', submission.url)
                    continue  # skip non-accepted
                if submission.score < MIN_SCORE:
                    print('Submission below score threshold: (%d / %d)' % (submission.score, MIN_SCORE))
                    continue  # skip submissions that haven't even reached 100 (thought this should be rare if we're collecting the "hot" submission)
                
				# Below: Check for the case of each host, download and save image. 

                if 'imgur' in submission.url:
					# Imgur submission
                    print('Imgur submission')

                    image_url, local_file_name, img_id = None, None, None
                    if 'http://imgur.com/a/' in submission.url:
                        # This is an album submission.
                        # Skip
                        print('Skipping album submission')
                        continue
                    elif 'http://i.imgur.com/' in submission.url:
                        # The URL is a direct link to the image.
                        mo = imgur_url_pattern.search(
                            submission.url)  # using regex here instead of BeautifulSoup because we are pasing a url, not html

                        imgur_filename = mo.group(2)
                        if '?' in imgur_filename:
                            # The regex doesn't catch a "?" at the end of the filename, so we remove it here.
                            imgur_filename = imgur_filename[:imgur_filename.find('?')]

                        img_id = '.'.join(imgur_filename.split('.')[:-1])
                        image_url = submission.url
                    elif 'http://imgur.com/' in submission.url:
                        # This is an Imgur page with a single image.
                        html_source = requests.get(submission.url).text  # download the image's page
                        soup = BeautifulSoup(html_source)
                        image_url = soup.select('.image a')[0]['href']
                        if image_url.startswith('//'):
                            # if no schema is supplied in the url, prepend 'http:' to it
                            image_url = 'http:' + image_url
                        # image_id = image_url[image_url.rfind('/') + 1:image_url.rfind('.')]

                        if '?' in image_url:
                            image_file = image_url[image_url.rfind('/') + 1:image_url.rfind('?')]
                        else:
                            image_file = image_url[image_url.rfind('/') + 1:]

                        img_id = '.'.join(image_file.split('.')[:-1])

                    img_split = submission.url.split('/')
                    file_ext = img_split[-1].split('.')[1]

                    img_name = '%s_%s.%s' % (img_id, IMGUR, file_ext)
                    savepath = os.path.join(folder, img_name)
                    print(savepath)
                    if savepath in current_imgs:
                        print('Image already exists:', savepath)
                        continue

                    save_count += 1
                    print('Date submitted:', get_date(submission))
                    # current_imgs.append(savepath) ???
                    download_image(image_url, img_name, folder=folder, id=i)
                elif any([host in submission.url for host in allowed_hosts]):
					# non-Imgur submission
					
                    img_split = submission.url.split('/')
                    img_id, img_name = None, None
                    if 'cdn.awwni.me' in img_split:
                        print('cdn.awwni.me submission')
                        img_id, file_ext = img_split[-1].split('.')
                        img_name = '%s_%s.%s' % (img_id, CDN_AWWNI_ME, file_ext)
                    elif 'i.redd.it' in img_split:
                        print('i.redd.it submission')
                        img_id, file_ext = img_split[-1].split('.')
                        img_name = '%s_%s.%s' % (img_id, REDDIT, file_ext)
                    else:
                        print(i, 'Unknown host:', submission.url)

                    savepath = os.path.join(folder, img_name)
                    print(savepath)
                    if savepath in current_imgs:
                        print('Image already exists:', savepath)
                        continue

                    save_count += 1
                    print('Date submitted:', get_date(submission))
                    download_image(submission.url, img_name, folder=folder, id=i)
                else:
                    print(i, 'Unknown host:', submission.url)

            except IndexError as ie:
                print(ie)
                print('IndexError trying to parse:', submission.url)
            except Exception as ex:
                print(ex)
                print('Unknown Exception trying to parse:', submission.url)


    def get_date(submission):
        time = submission.created
        return datetime.datetime.fromtimestamp(time)

    def collect_images(limit=100, daterange=None):
        r = get_client()
        if daterange is None:
            print('Collecting images based on limit of:', limit)
            submissions = r.subreddit(target_subreddit).hot(limit=limit)
        elif isinstance(daterange, tuple):
            print('Collecting images in date range:', (start_unix, end_unix), 'with limit:', limit)
            submissions = r.subreddit(target_subreddit).submissions(start_unix, end_unix)
        else:
            raise ValueError('submissions should be a tuple of (start, end) unix '
                             'or None to get currently hot.')

        find_images(submissions, limit=limit)

    collect_images(limit=limit, daterange=daterange)

##################################################

def detect_pics(folder=IMAGE_FOLDER):
    filetypes = ('.png', '.jpeg', '.jpg', '.tiff', '.bmp')
    files = []
    for filetype in filetypes:
        files.extend(glob.glob(os.path.join(folder, '*%s' % filetype)))

    return files

def load_image_PIL(filename):
    img = Image.open(filename)
    # img.show()
    return img

def save_image_PIL(img, filename, savedir=RESIZE_FOLDER, id=0):
    savepath = os.path.join(savedir, filename)
    img.save(savepath)
    print(id, 'Successfully saved image:', filename)

##################################################

# Crop images to 256 x 256
def crop_256px_pics(sq_res=resize_dim, folder=IMAGE_FOLDER):
    files = detect_pics(folder=folder)
    resized_files = set(detect_pics(folder=RESIZE_FOLDER))
    print(files)
    print('Number of existing files:', len(resized_files))
    
    for i, filename in enumerate(files):
        path = '\\'.join(filename.split('\\')[1:])
        savepath = os.path.join(RESIZE_FOLDER, path)
        
        if savepath in resized_files:
            print(i, 'Image already saved:', savepath)
            continue  # we've already downloaded files for this reddit submission

        try:
            img = load_image_PIL(filename)
        except OSError as ose:
            print(ose)
            print('Image couldn\'t be loaded - perhaps ran into 404 when downloading')
            continue
        except Exception as ex:
            print(ex)
            print('Unknown exception trying to load image:', filename)
        
        try:
            width, height = img.size
        except UnboundLocalError:
            print('Uncaught error from before? File:', filename)
            continue

        if min(width, height) < max(sq_res):
            # Image resolution is too low
            continue
        min_dim = None
		
		# Crop the image by taking middle square portion if width > height,
		# or by taking topmost square portion if height > width.
        try:
            if width < height:
                min_dim = width
                ratio = height / width

                # Crop TOP portion of image
                img = img.crop((0, 0, min_dim, min_dim)).resize(sq_res, Image.ANTIALIAS)
            elif height < width:
                min_dim = height
                ratio = width / height

                # Crop MIDDLE portion of image
                mid = width // 2
                img = img.crop((mid-min_dim//2, 0, mid+min_dim//2, min_dim)).resize(sq_res, Image.ANTIALIAS)
            else:
                img = img.resize(sq_res, Image.ANTIALIAS)
                ratio = 1
        except Exception as ex:
            print(ex)
            print('Error cropping image:', filename)
            continue

        if i > 1:
            pass

        try:
            save_image_PIL(img, path, savedir=RESIZE_FOLDER, id=i)
            resized_files.update([os.path.join(RESIZE_FOLDER, path)])
        except Exception as ex:
            print(ex)
            print('Error saving image:', filename)


if __name__ == '__main__':.
	# Collect images then crop 256 x 256 sections using these:
	
    # collect_anime_pics(limit=6000, daterange=(start_unix, end_unix))
    # crop_256px_pics()
	
	pass