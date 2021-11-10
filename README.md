# Neural Computation CourseWork 2021

Neural Computation\
Coursework 2021/2022\
Released: Friday November 5th 2021\
Deadline: Monday November 29th 2021 at 18:00 (UK time) for all students.\
Weight: 20 %

Group 31\
Team mebmers: Joshua Foulkes, Atharv Garg, Constantinos Iakovou, Nicoletta Lambrou, Katerina Michalaki, Aparna Nadaraj, Mark Suraj

Read the description of the assinment, link is here: https://canvas.bham.ac.uk/courses/56310/assignments/335411

## Getting Started
1. Go to Canvas Assignment and download CW.zip. Unzip the files to your home directory
(for example it is /Users/jduan/Desktop/CW on my laptop). You should then see data and
tutorial.ipynb in the unzipped folder.

2. After these files are downloaded and unzipped it is fine you develop your method using Jupyter
Notebook on your own laptop, but you will need to install torch and opencv python libraries.
Alternatively, we recommend you use Docker where we have installed all dependencies for you.
Open the terminal and type the following two command lines

. docker pull bhamcs/nc21cw \
. docker run -it --rm -v /Users/jduan/Desktop/CW:/src -p 8888:8888 bhamcs/nc21cw

The only thing you need to change above is to replace /Users/jduan/Desktop/CW with the
directory on your laptop (for Mac and Linux users only). If you are a Windows user, you
should replace the second command above with\

. docker run -it --rm -v C:\Users\jduan\Desktop\CW:/src -p 8888:8888 bhamcs/nc21cw\

Afterwards, copy the link displayed in the terminal onto your browser to start Jupyter Notebook from Docker, then click on tutorial.ipynb and complete the tutorial.
Important notes: \
(1) the Docker use here is slightly different to lab tutorials. Please follow
the instructions here only;\
(2) because I use -v parameter to pass /Users/jduan/Desktop/CW
on my local laptop to /src in Docker, all files under /Users/jduan/Desktop/CW will be
uploaded to /src in Docker. It is important you remember to use Docker paths (ie. /src) all
the time in your code development;\
(3) these two paths are synchronized. That is to say if
you change anything in /Users/jduan/Desktop/CW the content in /src will be automatically
updated and vice versa. In other words, you do not need to worry about losing your work
if you terminate Docker Jupyter Notebook and you have all your work saved locally on your
laptop.

3. After your network is trained deploy it on the images in the test set in /src/data/test/images
and save the predicted segmentation masks to /src/data/test/masks. If successful the masks
will be also produced in /Users/jduan/Desktop/CW/data/test/masks due to synchronization. You then need to run the submission conversion code in the last part of tutorial.ipynb
to encode your masks into a CSV file and then submit it to Kaggle for ranking. We have
provided a video to explain how the process is done exactly.

## Report
We are also require to write a report on the coursework.\
We will start writing our report after we finish we the source code.

Write one group report explaining what you did and what you found. Within the report you
should insert source code. A reasonable length for the report would be between 2000 and
3000 words excluding the reference list, plus as many diagrams, tables and graphs as you
think appropriate. The 20 marks are divided into four sections: training (30%), validation
(10%), inference (10%), code quality (10%) and report quality (40%). The first four sections
are marked based on your inserted source code and the last section on your written report.

- For training, you are expected to show the following components: network architecture, loss function, optimizer and training processing. For the training process, you should show your understanding on the number of epochs required to train your network, as well as data loading. It also necessary to show correct training mode (model.train), zero gradient (optimizer.zero grad), backpropagation (loss.backward), optimization (optimizer.step), etc.
- For validation, you are expected to show how you use this process to select reasonable hyperparameters in your network.
- For inference (deployment), you are expected to show how well your trained model has performed on test set on Kaggle. The performance will account for 50% and the remaining 50% comes from your inference code. Ideally, you should save the trained model and then load it for inference.
- For coding, high quality code should be well structured and commented.
- For the report, marks are further divided into
    - Introduction (10%): discuss the data sets involved, the machine learning task, relevant work and what you aimed to achieve.
    - Implementation (35%): describe how you implemented your neural network and the associated performance analysis mechanisms. Explain why you chose to do it that way. Remember to cite any sources you used.
    - Experiment (45%): describe the experiments you carried out to optimize your network’s generalization performance and present the results you obtained. Explain in detail how you used the training, validation and test data sets. The results should be presented in a statistically rigorous manner.
    - Conclusion (10%): summarize your key findings, including which factors proved most crucial, and what was the best generalization performance you achieved.

## Submission
The group leader should submit on Canvas a HTML file titled group#.html, where # should be
replaced by the group number. More specifically, the HTML file is generated from your Jupyter
Notebook (File → Download as → HTML). In your Notebook, you should include source code and
write the report with the corresponding section headings as described above.

## Help and Advice
For any question feel free to ask in any of these two \
Link to MS Teams: https://teams.microsoft.com/l/team/19%3ab1btuDg2khJeh6bD8vuSfDrWeaNKSysmPYwDFW_1vvM1%40thread.tacv2/conversations?groupId=ecd0b4c5-4a0b-4c73-8f19-69a2c243d4ed&tenantId=b024cacf-dede-4241-a15c-3c97d553e9f3 \
Link to our Discord server: https://discord.gg/FkKCYdMG
## GitHub Help
### Making a commit
[git commit](https://www.atlassian.com/git/tutorials/saving-changes/git-commit)

Please use a sensible commit message that tells the rest of the team what change or feature has been added.

E.g. `git commit -m "trained the neural network"`.

Please make commits often. This ensures that the rest of the team can see the steps you've taken to implement
a new feature, and it makes it easier to reset to a previous version of your code.

### Pushing commits
[git push](https://www.atlassian.com/git/tutorials/saving-changes/git-commit)
