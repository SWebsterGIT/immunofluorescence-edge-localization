# immunofluorescence-edge-localization
This code represents a script for determining relative intensities of immunofluorescent staining in cells. Specifically, it determines the intensity in the overall cell in comparison to edge regions, in order to provide a quantitative measurement of protein translocation.

This script, implemented as a jupyter notebook (.ipynb), is a simple data analysis tool built for the specific purpose of research data analysis and is not intended to be used in production systems, or as a general tool beyond its original scope. 

# Instructions

Before getting started, download some of the demo images included in this github page. For example, to download NUMA-INFECTED.tiff, click on the file from the github repo, then select "Download raw file" as shown below:

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/372980db-1fbd-4bdd-988e-a6a4de2516a3)

For the following steps, you'll then want to upload the file to your google drive, in a place you can find it later.

This jupyter notebook was designed to be run in google colab. An instance of it for use is available here: [link](https://colab.research.google.com/drive/1E_3s3FWmBz4fHAkoZTWepFbNuTg_Ce7v?usp=sharing). 


The benefit to using colab is it allows users with different machines, operating systems, and python installations to use a common arena to run code. 

To get started, open the colab instance in your browser.

To make edits, you'll need to make a copy of the notebook, use "Save a copy in drive" to do this.

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/e9f7dcbf-6c3d-4aea-ad33-88e3cd769fbb)

Run the first section of the notebook by clicking the "play" icon next to it. This should trigger a prompt to allow access to google drive. This will be used to supply the images for processing.

Once drive access is in place, you'll need to specify the path the script should use to access the image you want to analyze. An easy way to get the path is to use the file folder icon in colab, locate the file in your drive, click on the 3 dots next to it and click "copy path". 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/da61459d-8d67-45b7-8a92-581234db20dc)


Paste this path into the jupyter notebook "path" line. It should appear something like -> path = '/content/drive/MyDrive/covid_images/demo_trial_1.jpg'

You may then choose a name for your image. This will only be used if you chose to download the resulting data as a csv file in the end. 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/5f2f69a4-21c3-4e74-b137-331766e2287a)


Now, update the NUM_COLONIES value based on how many distinct colonies you wish to perform analysis on. The script will prioritize them in order of size. So NUM_COLONIES = 2 will perform analysis on the largest and second largest distinct colonies present in the image.

Now, in the menu bar of colab, select Runtime -> Run All. 

Running a typical analysis takes on the order of 30 seconds under normal conditions. 

To view the results of the analysis, scroll down to the bottom of the notebook and view the generated graphs. The expected results for Numa-infected, for example, are: 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/411fe334-93a2-4ad4-b70d-623c813a16ef)


As you scroll down, you will also be presented with a copy of the original image you input, which you may use to confirm you have run analysis on the correct image. For example: 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/c75c6f2d-4976-4771-9293-f1aff8cc147b)


Images showing an overlay of the regions that were used in analysis are also produced. These images can be used to manually visually confirm the correct analysis was run. For example, the below image shows the interior region of Numa-Infected, which we can visually verify.

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/fe501195-128f-4e33-a7c0-69c0790ecb1f)


If you wish to download the data from the images you've run analysis on as a csv file, one is generated and updated each time you run the script. It is located in the colab file system and can be downloaded by clicking the three dots and selecting "download." 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/c66f7a7a-0ce2-442e-83f7-b0df04e1f621)



If you wish to use this software tool for your own arbitrary images beyond those it was originally tuned for, certain parameters may be useful to adjust. In principle, these include: 
RADIUS_FROM_EDGE, which can be changed to alter the definition of the width of the edge of your colonies, measured in pixels. If your image is a different size or resolution this should definately be adjusted accordingly. 

SCALE_BAR_LENGTH_UM, which should be changed to reflect the length of your scale bar. Note that only white scale bars are currently supported, and it is benefecial to the software to have a scale bar that does not overlap with regions you wish to analyse. 

MAX_BRIDGE_RADIUS, which can be used to prevent the software from interpreting cut-off regions as edges. This is located and used in the function "bridge_gaps_on_edges".

"lower_bound" may be useful to change if your image is very dark or exceedingly bright. 

DEBUG_MODE can be set to True to output a more comprehensive set of intermediately processsed images, if needed. It may help to have an understanding of image processing software in order to use this, and the other parameters, most efficiently.


# Dependencies
OpenCV2 (python)
matplotlb (python)
numpy (python)
pandas (python) - for csv exports only
colab + google drive integration

The code is designed to run in a google colab jupyter notebook enviornment. Since colab runs using a virtual machine, it should be able to run on any machine capable of running colab in the browser.
All the dependencies should install via the "import" instructions in the code. 
These sections need to be run first in order for everything to work seamlessly- this is taken care of by running the sections in order. The current version of the code requires the images to be housed in a google drive with permission given to access them. The colab instance should prompt the user to allow access the first time that section of the notebook is run.

# License
This project is using the GPL ( GNU General Public License ). 
