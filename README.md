# immunofluorescence-edge-localization
This code represents a script for determining relative intensities of immunofluorescent staining in cells. Specifically, it determines the intensity in the overall cell in comparison to edge regions, in order to provide a quantitative measurement of protein translocation.

This script, implemented as a jupyter notebook (.ipynb), is a simple data analysis tool built for the specific purpose of research data analysis and is not intended to be used in production systems, or as a general tool beyond its original scope. 

# Instructions

This jupyter notebook was designed to be run in google colab. An instance of it for use is available here: link. 

The benefit to using colab is it allows users with different machines, operating systems, and python installations to use a common arena to run code. 

To get started, open the colab instance in your browser.

Run the first section of the notebook by clicking the "play" icon next to it. This should trigger a prompt to allow access to google drive. This will be used to supply the images for processing.

Once drive access is in place, you'll need to specify the path the script should use to access the image you want to analyze. An easy way to get the path is to use the file folder icon in colab, locate the file in your drive, click on the 3 dots next to it and click "copy path". Paste this path into the jupyter notebook "path" line. It should appear something like -> path = '/content/drive/MyDrive/covid_images/demo_trial_1.jpg'

You may then choose a name for your image. This will only be used if you chose to download the resulting data as a csv file in the end. 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/5f2f69a4-21c3-4e74-b137-331766e2287a)


Now, update the NUM_COLONIES value based on how many distinct colonies you wish to perform analysis on. The script will prioritize them in order of size. So NUM_COLONIES = 2 will perform analysis on the largest and second largest distinct colonies present in the image.

Now, in the menu bar of colab, select Runtime -> Run All. 

Running a typical analysis takes on the order of 30 seconds under normal conditions. 

To view the results of the analysis, scroll down to the bottom of the notebook and view the generated graphs. 

As you scroll down, you will also be presented with a copy of the original image you input, which you may use to confirm you have run analysis on the correct image.

Images showing an overlay of the regions that were used in analysis is also produced. These images can be used to manually visually confirm the correct analysis was run. 


If you wish to download the data from the images you've run analysis on as a csv file, one is generated and updated each time you run the script. It is located in the colab file system and can be downloaded by clicking the three dots and selecting "download." 

![image](https://github.com/SWebsterGIT/immunofluorescence-edge-localization/assets/90474441/c66f7a7a-0ce2-442e-83f7-b0df04e1f621)



# Dependencies
OpenCV2 (python)
matplotlb (python)
numpy (python)
pandas (python) - for csv exports only
colab + google drive integration

The code is designed to run in a google colab jupyter notebook enviornment. Since colab runs using a virtual machine, it should be able to run on any machine capable of running colab in the browser.
All the dependencies should install via the "import" instructions in the code. 
These sections need to be run first in order for everything to work seamlessly- this is taken care of by running the sections in order. The current version of the code requires the images to be housed in a google drive with permission given to access them. The colab instance should prompt the user to allow access the first time that section of the notebook is run.
