# superfluid_simulator
Hello! Please note that this description is **unfinished**, please come back soon (unless you actually want to read my uncurated insanity).

An exploration into the mechanics of Helium-4 and replicating these properties in a real-time simulation for use in education. 

Description to be added here as well as operating instructions, testing videos to be uploaded as examples.

Firstly, I'd like to mention a few of the projects I used as inspiration, and as help for when my tiny brain couldn't cope with the science:

[WebGL Superfluid Simulation using dGPE (George Stagg)](https://georgestagg.github.io/webgl_gpe/) - this gave me a lot of the initial inspiration for the idea. 

[QMSolve](https://github.com/quantum-visualizations/qmsolve) - Linear Schrödinger equation solver, this was a massive, massive help on the animation routine (some of mine is heavily based off this project).

I also got some help in wrapping my head around the dGPE from [here](https://github.com/TarkhovAndrei/DGPE).

Below is detailed a simplified and concise guide to running this simulation - if you would like to see the full report and mathematical/scientific detail, you'll find attached a PDF document which goes into 112 pages of painstaking detail.

### Prerequisites
In order to run this simulation, you will require Python 3. The version in which I created and tested this project in was 3.11, although I've tried both 3.10 and 3.12 and found no issues. If you encounter a problem running this project, please try and use an environment with 3.11 installed to prevent compatibility issues. It's best to set up a virtual environment, it will also make the next part of the setup much simpler.
Included in the repo is a requirements.txt file, which includes all the dependencies that are required to run this project, as well as some extras which I experimented with whilst trialling different maths and animation libraries. You can download this file to the project directory and use the command `pip install -r /path/to/requirements.txt` to install all of these dependencies automatically. I've marked the optional modules with a comment in the file.

### Using the simulation 
When run, the project will (should) greet you with a friendly, if mildly ancient GUI window asking a lot of difficult questions. I've included the paramaters needed for some of the cool examples below, but I advise you play around with the parameters (assuming you are the singular person who has made it this far).

-- picture of GUI --

Once you have loaded in the desired parameters, click `Create Simulation`. This will quite literally create a simulation object. When this is done, a button for running the simulation will appear, aptly labelled `Run Simulation`. Clicking this will close the dialog window, and status updates will appear in the terminal (more to alliviate my panic than for any genuine need). Once the program reaches the SSFT stage, a progress bar will appear giving an estimated time and percentage progess through the evolution routine. Depending on your selection of timestep width and total duration, this can take quite some time (more specifically, with reasonable arguments expect this to take anywhere between 30 seconds and 6 minutes). 

-- sc of terminal -- 

Finally, a new window should appear with the simulation inside (along with a new dialog box to run another simulation). Hopefully contained within this new window is a plot area containing the desired animation. There are some basic matplotlib tools included, as well as a few useful functions of my own on the side pane. I think these are relatively self-explanatory, but if you do require assistance, lots of detail is included on these (as well as specifics on their implementation and limitations) within the PDF report.

-- sc of simulation window -- 

### Cool examples
With the basics out of the way, we can get to the true purpose of such simulations - rainbows!

-- insert extremely cool example of rainbow tracing --


-- and maybe one actually useful example -- 


https://github.com/user-attachments/assets/7e3bbe5c-2387-425b-858f-9730563ce265

