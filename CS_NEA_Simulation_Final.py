# Ismail Mehmood - Simulating a Helium-4 boson in 2D real space 
# 
### Imports
from abc import ABC, abstractmethod
# abc (Abstract Base Class) library used to make abstract base class for the particle system
import numpy as np
# numpy is used for the following: zeros, dtype, pi, exp, linspace, meshgrid, amin, amax, round, abs, fft.fftn, fft.ifftn, array, ndarray, angle, moveaxis, where, concatenate
from tkinter import *
# tkinter is the python interface to the Tk GUI toolkit. This is used for StringVar, Label, Button, OptionMenu, focus_set, pack, Frame, title, geometry, get, set.
import matplotlib.pyplot as plt
# Used for plotting program output: 
from matplotlib import widgets
# Used to create the toolbar 
from matplotlib import animation
# Used to animate the frames of the simulation generated from the plot routine.
from matplotlib.colors import hsv_to_rgb
# module from matplotlib which converts the HSV colours obtained from the complex components of the energy matrices to RGB.
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
# this backend plugin is used to integrate the plot from matplotlib into the tkinter GUI.
from tqdm import tqdm
# tqdm is a wrapper for iteratives which generates a progress bar. This gives the user an indication of the time taken for the time-dependent solver routine.
#import ffmpeg
# video library for handling video (used to download animation)
from scipy import ndimage # integrate, sparse
# scipy is used for the following: ndimage (for the laplace integral routine to approximate the second spatial derivative) Previously used integrate (to calculate the normalisation constant of the wavefunction) and sparse (to create a sparse 2D array from a matrix)
from functools import partial
# creates a partial function from a function and parameters in order to package parameters into tkinter buttons

### Physical contants (to 11dp)

# Four fundamental physical constants
# note, these constants are in Hartree atomic units for simpler and faster calculations. 
hbar = 1.0 # reduced Plank's constant
e = 1.0 # elementary charge
k_e = 1.0 # Coulomb constant
m_e = 1.0 # electron rest mass

# Interaction Potential of Helium-4 boson
a_s = 1.6782483e-9
#a_s = 1.0
# Time constants - ***seconds to atomic time unit (hbar/hartree energy)
picoseconds = 4.134137333518212e4
femtoseconds = 4.134137333518212 * 10.

# each constant here is to 17sf
Å = 1.8897261246257702 # angstrom
m_p = 1836.1526734400013

k = hbar**2 / (2*m_e) # force constant for oscillator.

# Classes for particle system, simulation and time evolution:
# Step 2: Define the objects


class GPE:
    def __init__(self, gridpointtuples, H):
        #spatialext is a list of tuples
        self.H = H
        self.gridpoint = H.N
        if len(gridpointtuples) == 2:
            self.nX, self.nY = gridpointtuples
        elif len(gridpointtuples) == 4:
            self.nX, self.nY, self.nXX, self.nYY = gridpointtuples
        self.create_wfc_arrays()

    def create_wfc_arrays(self):
        if type(self.H.particle_system) == singleparticle:
            self.wfc = np.zeros((self.nX, self.nY), dtype = np.complex128) # wavefunction for real space
        elif type(self.H.particle_system) == twoparticles:
            self.wfc = np.zeros((self.nX, self.nY, self.nXX, self.nYY), dtype = np.complex128)
    
    def wfc_0(self, x1, y1): # initial wavefunction initialiser, takes in an x and y and outputs a function of the two representing the wavefunction from that position
        eps1 = 0.25
        const = 8**0.25/(np.pi * eps1)**(3/4)
        return const * np.exp(-(x1**2 + 2*y1**2/(2*eps1)))
    
    def wfcd_0(self, x1, y1, x2, y2): # initial wavefunction initialiser for dual particle systems, outputs a function of the 4 xy parameters representing the wavefunction from that position.
        #This wavefunction correspond to two stationary gaussian wavepackets. The wavefunction must be symmetric: Ψ(x1,x2) = Ψ(x2,x1)
        σ = 0.4 * Å
        𝜇01 = -7.0*Å
        𝜇02 = 0.0*Å
        return (np.exp(-(x1 - 𝜇01)**2/(4*σ**2))*np.exp(-(x2 - 𝜇02)**2/(4*σ**2)) + np.exp(-(x1 - 𝜇02)**2/(4*σ**2))*np.exp(-(x2 - 𝜇01)**2/(4*σ**2))) + (np.exp(-(y1 - 𝜇01)**2/(4*σ**2))*np.exp(-(y2 - 𝜇02)**2/(4*σ**2)) + np.exp(-(y1 - 𝜇02)**2/(4*σ**2))*np.exp(-(y2 - 𝜇01)**2/(4*σ**2)))

    def calc_normalisation_constant(Hm, psi): # normalisation constant of wavefunction
        n_c = Hm.dV * (np.sum(np.abs(psi)))**2 # The probability of the particle being in the real space is 1.
        return n_c
    
    def renormalise(Hm, normconstant, wfc): # normalisation routine - tales in the calculated normalisation constant and divides wfc by it
        wfc /= normconstant
        return wfc

    def initialsinglewavefunction(self, Hm): # defines each position in the wfc array as a function from wfc_0
        self.wfc[:] = self.wfc_0(Hm.particle_system.x1, Hm.particle_system.y1) # the colon fetches the entire array
        print("wavefunction:", self.wfc) # WB 3.1.3
        return self.wfc
    
    def initialdualwavefunction(self, Hm): # defines each position in the wfc array as a function from wfc_0
        self.wfc[:] = self.wfcd_0(Hm.particle_system.x1, Hm.particle_system.y1, Hm.particle_system.x2, Hm.particle_system.y2) # the colon fetches the entire array
        return self.wfc


# The class particle_system is used for the object representing the particle system. To allow for further development, the particle_system class will have methods only general to all particle systems.

class particle_system(ABC): # abstract base particle class, this is used to pass the methods into each particle system.
    def __init__(self):
        pass
    
    @abstractmethod
    def get_system_observables(self, H): # creates the 1D vectors and meshgrid to form a linear space.
        pass
    
    
# The class singleparticle is used for the object representing any single particle system. It requires the mass of the particle and the spin to instantiate, but will assume the mass to be m_e and the spin to be None if not provided.
# Methods here are: get_system_observables - x

class singleparticle(particle_system):
    def __init__(self, m=m_e): # this initialiser assumes a single particle has the mass of an electron if left blank.
        self.m=m # mass of particle to be modelled

    def get_system_observables(self, H): # returns a meshgrid which functions as a composition of the linear spaces needed to represent the system
        #create meshgrid of space equivalent to hamiltonian based on 2D
        x1 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        y1 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        self.x1, self.y1 = np.meshgrid(x1, y1) # The meshgrid is a co-ordinate matrix of the numpy arrays provided.
        print(self.x1) #WB 1.2.2
    
# The class twoparticle is used for the object representing any single particle system. It requires the mass of the particle to instantiate, but will assume the mass to be m_e if not provided.

class twoparticles(particle_system):
    def __init__(self, m=m_e):
        self.m=m # mass

    def geteigenstates():
        return
    
    def get_system_observables(self, H): # returns a meshgrid which functions as a composition of the linear spaces needed to represent the system
        #create meshgrid of space equivalent to hamiltonian based on 2D - 4 1D arrays for 2D, each.
        x1 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        y1 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        x2 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        y2 = np.linspace(-H.spextent/2, H.spextent/2, H.N)
        self.x1, self.y1, self.x2, self.y2 = np.meshgrid(x1, y1, x2, y2) # The meshgrid is a co-ordinate matrix of the numpy arrays provided.
        #print(self.x1) #WB 1.2.2


# The class Hamiltonian represents the operator object corresponding to the total energy of the system. The Hamiltonian is a collation of the informtion needed to create a wavefunction independent of the particle position.

class Hamiltonian:
    def __init__(self, particles, potential, pot_time, pot_amp, gridpoints, spextent, spatial_ndim, E_min=0):
        self.N = gridpoints # number of gridpoints to use
        self.spextent = spextent # spatial extent of the system, in angstroms
        self.dx = self.spextent / self.N # space width, estimate for spatial derivative operator for x
        self.dy = self.spextent / self.N # space width, estimate for spatial derivative operator for y
        self.dV = self.dx * self.dy # 2D derivative operator
        self.particle_system = particles # particle system being used
        self.particle_system.H = self # the Hamiltonian of the particle system, for easy reference
        self.spatial_ndim = spatial_ndim # why?
        self.ndim = 4  # total number of observables
        self.potential = potential # assigns the potential type
        self.pot_time_period = pot_time # time period of harmonic oscillator, if present
        self.pot_max_amp = pot_amp # max amplitude of harmonic oscillator, if present

        # The following method calls are part of the constructor.

        self.particle_system.get_system_observables(self) # initialise the obervables, uses a particle method

        self.V = self.getpotentialasamatrix() # gets the potential in a matrix form in order to 

    def getpotentialasamatrix(self): # converts potential into a matrix
        V = self.potential(self.particle_system, self.pot_time_period, self.pot_max_amp) # gets the potential from the global method as defined by the type of potential required.
        self.Vgrid = V # The global potential routine returns the potential as a function of the particle position in space, which is stored as a grid.
        self.E_min = np.amin(V) # determines the minimum energy value - needed to animate the function
        if type(self.particle_system) == singleparticle:
            V = V.reshape(self.N, self.N) # shapes the potential into a N^2 point matrix
        elif type(self.particle_system) == twoparticles:
            V = V.reshape(self.N, self.N, self.N, self.N) # shapes the potential into a N^4 point matrix
        print(V) #WB 2.2.1
        return V # returns the generated matrix in an array form
    
    
# The class Simulation takes in the time parameters to initialise the time conditions for the simulation.

class Simulation:
    def __init__(self, H, total_time, store_steps):
        # simple creation of attributes assigned to object based on input parameters
        self.dt  = total_time/store_steps # width of time step
        self.total_time = total_time # total time to run simulation
        self.store_steps = store_steps # steps to store
        self.H = H # associated Hamiltonian
        print(self.dt, " - dt, ", self.total_time, " - total time, ", self.store_steps, " - steps to store.") # WB 4.1.1
        
    def sim(self, psi, Hmt): # The sim method takes in a wavefunction and a Hamiltonian to add to the simulation object and evolve using a specified method. In this case, the only method is the split_step_fourier.
        time_evolution = split_step_fourier(self) # instantiates an object of type split_step_fourier
        self.results = time_evolution.run(psi, Hmt) # uses the run() method of the time evolution object created, passing in an initial wavefunction and a Hamiltonian to process the time evolution.
        print(self.results[0], self.results[int(self.store_steps/2)], self.results[self.store_steps], " - evolved wfcs at t = 0, t = t/2 and t = t") # WB 4.2.1
        return self.results # returns the array storing the calculated wavefunctions.


# The class split_step_fourier is a method to carry out the time evolution of the simulation. It takes in a simulation object and evolves the associated wavefunction for the number of timesteps required.

class split_step_fourier:
    def __init__(self, simulation):
        self.simulation = simulation # assigns simulation as an attribute
        self.simulation.Vmin = np.amin(self.simulation.H.Vgrid) # finds minimum potential
        self.simulation.Vmax = np.amax(self.simulation.H.Vgrid) # finds maximum potential
        
    def run(self, wfc, Hm): # carries out the time evolution
        self.Hm = Hm # assigns the Hamiltonian attribute
        steps_to_make = self.simulation.store_steps # takes in the number of steps to carry out from the simulation
        dt = self.simulation.dt # takes in the step width to use from the simulation
        total_time = self.simulation.total_time # takes in the total time to evolve over from the simulation
        dt_store = total_time/dt # the number of steps to take by width
        dt = dt_store # time step width
        psi = [wfc] # import initial wavefunction
        c_0 = 1 # constant for split-operator approximation to GPE
        c_1 = -c_0 # constant for split-operator approximation to GPE
        c_2 = 1 # constant for split-operator approximation to GPE
        g = 4*np.pi*hbar**2*a_s/self.Hm.particle_system.m # coupling constant
        potential = self.Hm.V # takes the potential matrix from the Hamiltonian

        for i in tqdm(range(0, steps_to_make), desc = "Progress (of steps taken): "): # tqdm wrapper for progress bar + iteration for number of steps to carry out
            TPSI = [] # temporary array to store wavefunctions being manipulated
            TPSI.append(psi[i]) # add the current array psi into TPSI

            operatorone = np.exp(-0.5j*dt*(potential+(g*np.abs(TPSI[0]**2)))) # first operator, for position space
            operatortwo = np.exp(0.5j*dt)*(ndimage.laplace(TPSI[0])) # second operator, for momentum space

            TPSI.append(operatorone*(np.fft.fftn(TPSI[0]))) # TPSI[1] is psi_1, the first wavefunction from the partial operator application, here psi is stored in momentum space using the fft.fftn n-dimensional fourier transform
            if i == 1:
                print(TPSI[1], " - step 1 for Test 5.2.1a") # WB Test 5.2.1

            TPSI.append(operatortwo*(np.fft.ifftn(TPSI[1]))) # TPSI[2] is psi_2, the second wavefunction from the partial operator application, here psi is stored in real space using the fft.ifftn n-dimensional inverse fourier transform
            if i == 1:
                print(TPSI[2], " - step 2 for Test 5.2.1b") # WB Test 5.2.1
            operatorthree = np.exp((-0.5j*dt*(potential+g*np.abs(c_0*TPSI[0]+c_1*TPSI[1]+c_2*TPSI[2])**2))) # the third operator is now built - as it uses psi_1 and psi_2 as part of it's definition, the first two operations had to be carried out first.

            TPSI.append(np.fft.ifftn(operatorthree*(TPSI[1]))) # the third operator can now be applied to the psi_1 wavefunction
            if i == 1:
                print(TPSI[3], " - step 3 for Test 5.2.1c") # WB Test 5.2.1
            norm = GPE.calc_normalisation_constant(self.Hm, TPSI[3]) # uses the GPE method to calculate a normalisation constant to the wavefunction.
            final_psi = GPE.renormalise(self.Hm, norm, TPSI[3]) # uses the GPE method to normalise the final wavefunction
            if i == 1:
                print("Normalisation constant: ", norm) # WB 3.2.1
                print(self.Hm.dV * (np.sum(np.abs(final_psi)))**2, " - area under normalised wavefunction, should be 1.") # WB 3.2.1
            psi.append(final_psi) # appends the final wavefunction to the psi array
            
        self.simulation.psi_max = np.amax(np.abs(psi)) # finds the maximum of psi (for plot)
        return psi # returns psi - the array full of stored steps
        
# Method for defining the potential, here a potential correlating to a harmonic oscillator is used.
        
def harmonicoscillator(particle_system, pot_time, pot_amp):
    T = pot_time * femtoseconds # time period, Enter a suitable time period for the harmonic oscillator in femtoseconds
    w = np.pi*2/T # omega
    A = pot_amp * Å / femtoseconds * w # max magnitude of the oscillation
    if type(particle_system) == singleparticle:
        PE = 0.5 * A**2 * particle_system.x1**2 + 0.5 * A**2 * particle_system.y1**2 # total potential energy from position in harmonic oscillator
    elif type(particle_system) == twoparticles:
        PE = 0.5 * A**2 * particle_system.x1**2 + 0.5 * A**2 * particle_system.y1**2 + 0.5 * A**2 * particle_system.x2**2 + 0.5 * A**2 * particle_system.y2**2
    return PE
# In order to define the potential of the grid, we use the particle mass, time period (if oscillatory), omega. k is m*omega².
# This subroutine will return the energy due to a particles position in a harmonic oscillator.


# The object visualise_wavefunction contains methods to plot the simulation data using matplotlib in a tkinter GUI.

class visualise_wavefunction():
    def __init__(self):
        self.pause_status = False # pause status of animation
        self.restart = False # restart status of animation
        self.mpeg_write_status = False # video writing status of animation
        self.download_count = 0
        print(self.pause_status, " - pause status, ", self.restart, " - restart status, ", self.mpeg_write_status, " - video writing status, ", self.download_count, " - download count.") # WB 6.1.1

    def complex_to_rgb(self, Z: np.ndarray, max_val: float = 1.0) -> np.ndarray:
        mod = np.abs(Z) # takes modulus of complex value
        arg = np.angle(Z) # takes argument of complex value
        h = (arg + np.pi)  / (2 * np.pi) # hue value determined by polar mapping of argument
        #s = np.reshape((r/self.simulation.psi_max), (256, 256)) - this is the precise mapping of the colour space, but it leads to washed out colours and too low values of saturation to see clearly.
        #v = np.reshape((r/self.simulation.psi_max), (256, 256))
        s = np.ones(h.shape) # fills s, v with the value 1 - which is not colour accurate but for the purposes of the simulation provides a much higher quality output.
        v = np.ones(h.shape)
        rgb = hsv_to_rgb(np.moveaxis(np.array([h,s,v]) , 0, -1)) # moves h, s, v into an array, rearranges the axis and uses the hsv_to_rgb routine from matplotlib to convert the HSV value obtained into an RGB value.

        abs_z = mod / max_val # Takes the wavefunction saturation from the function call and sets the maximum value of the magnitude of the complex value to the 
        abs_z = np.where(abs_z> 1., 1. ,abs_z) # where the magnitude of the complex number is greater than 1, replace abs_z with 1
        return np.concatenate((rgb, abs_z.reshape((*abs_z.shape,1))), axis= (abs_z.ndim))
            
    def animate(self, figuresize=(10, 10), animation_duration = 5, fps = 30, potential_saturation=0.8, wavefunction_saturation=0.8):
        total_frames = int(fps * animation_duration) # total number of frames required
        sim_dt = self.simulation.total_time/total_frames # time to display each frame for
        self.simulation.psi_plotarea = self.simulation.results/self.simulation.psi_max # makes a suitable plot area as an attribute of the simulation
        graphfig = plt.figure(figsize=figuresize) # creates a figure
        axes = graphfig.add_subplot(111) # adds axes to the figure as part of a subplot arrangement 
        plt.style.use("dark_background") # background colour of simulation graph
        


        length = self.simulation.H.spextent/Å # length of system
        # plot of the potential on the axes of the potential grid over the range of the potential system, using a grey colourmap and using bilinear interpolation into
        plot_potential = axes.imshow((self.simulation.H.Vgrid + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-length/2, length/2, -length/2, length/2]) 
        # plot of the wavefunction
        plot_wavefunction = axes.imshow(self.complex_to_rgb(self.simulation.psi_plotarea[0], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent = [-length / 2, length/ 2,-length / 2,length / 2])
        print(self.complex_to_rgb(self.simulation.psi_plotarea[0], max_val= wavefunction_saturation), " - initial RGB of wavefunction") # WB 6.2.1

        axes.set_title("Dispersion of Helium-4 boson in R^2, psi(x, y, t)") # axes title
        axes.set_xlabel("x, angstroms") # x label
        axes.set_ylabel("y, angstroms") # y label
        
        #time label change
        time_ax = axes.text(0.96, 0.96, "",  color = "white", transform=axes.transAxes, ha="right", va="bottom")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % 0.00)) 
        
        animation_data = {'t': 0.0, 'ax':axes ,'frame' : 0}
        def func_animation(*arg): # increments the time, and calculates the next frame to display by calculating a frame for each time step, sim_dt. Used for the FuncAnimation call
            if self.restart == True:
                self.restart = False
                animation_data['t'] = 0.0 # reset the animation time to 0
            if not self.pause_status: # stops next frame being calculated when pasued
                time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (animation_data['t']/femtoseconds))) # update time axis value

                animation_data['t'] = animation_data['t'] + sim_dt # increment time
                if animation_data['t'] > self.simulation.total_time:
                    animation_data['t'] = 0.0 # replays simulation when toal time reached

                animation_data['frame'] +=1 # increments frame number
                index = int((self.simulation.store_steps)/self.simulation.total_time * animation_data['t']) # t/dt 

                plot_wavefunction.set_data(self.complex_to_rgb(self.simulation.psi_plotarea[index], max_val= wavefunction_saturation)) # the potential of the wavefunction is plotted over the plot area
                return plot_potential, plot_wavefunction, time_ax
            else:
                return plot_potential, plot_wavefunction, time_ax


        # call to animation function, takes in figure, animation function, frames and the interval to display the frame for. Uses blitting, a procedure outlined in the design for the animation function.
        self.a = animation.FuncAnimation(graphfig, func_animation, blit=True, frames=total_frames, interval= 1/fps * 1000) # Call to FuncAnimation with the parameters: graphfig - the plot figure, func_animation - the user defined function describing how to animate the simulation
        # blitting - renders all non-changing points in the animation into a single background image and renders it in one go. 

        # creating the Tkinter canvas 
        # containing the Matplotlib figure 
        self.canvas = FigureCanvasTkAgg(graphfig, master = self.root)   
        self.canvas.draw() 
  
        # placing the canvas on the Tkinter window 
        self.canvas.get_tk_widget().pack() 
  
        # creating the Matplotlib toolbar 
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.topframe) 
        self.toolbar.update() 
  
        # placing the toolbar on the Tkinter window 
        self.canvas.get_tk_widget().pack() 
        
    def mpeg_write(self): # writes video to mped
        self.mpeg_write_status = not self.mpeg_write_status # set mpeg write status
        print(self.mpeg_write_status, " - updated mpeg write status") # WB 6.1.1
        self.ToggleRestartStatus()
        if self.pause_status:
            self.TogglePauseStatus()
        Writer = animation.writers['ffmpeg'] # use ffmpeg as the writer
        writer = Writer(fps=30, bitrate=1800) # write file using given parameters
        self.a.save(r"C:\Users\ismail\Videos\Simulation_Downloads\animation"+str(self.download_count)+".mp4", writer=writer) # write animation to file using defined writer
        self.download_count += 1
        print(self.download_count, " - updated download count")
        self.mpeg_write_status = not self.mpeg_write_status # unset mpeg write status
        print(self.mpeg_write_status, " - updated mpeg write status") # WB 6.1.1

    
    def TogglePauseStatus(self):
        self.pause_status = not self.pause_status # changes pause status
        print(self.pause_status, " - new pause status") # WB 6.1.1
         
    def ToggleRestartStatus(self):
        self.restart = not self.restart # changes restart status
        print(self.restart, " - restart status change") # WB 6.1.1

    
    def InputDialogue(self):
        self.inputwindow = Tk()
        self.inputwindow.geometry("450x600")
        self.inputwindow.title("Input Dialogue")
        grd_options = [ 
            "128", 
            "256", 
            "512", 
        ] 

        # datatype of menu text 
        grd_clicked = StringVar()
  
        # initial menu text 
        grd_clicked.set("128") 
  
        # Create Dropdown menu 
        grd_drop = OptionMenu(self.inputwindow, grd_clicked, *grd_options ) 
        grd_drop.pack() 
    
        # Create Label 
        self.gridpoints = Label(self.inputwindow, text = " " ) 
        self.gridpoints.pack()
  
        # Create button, it will change label text 
        grd_button = Button(self.inputwindow, text = "Update", command = partial(self.show, self.gridpoints, grd_clicked)).pack()

        ### NUMBER OF PARTICLES
    
        # Dropdown menu options 
        par_options = [ 
            "1", 
            "2 - Superposing", 
        ] 

        # datatype of menu text 
        par_clicked = StringVar()
  
        # initial menu text 
        par_clicked.set("1") 
  
        # Create Dropdown menu 
        par_drop = OptionMenu(self.inputwindow, par_clicked, *par_options ) 
        par_drop.pack() 

        # Create Label 
        self.particles = Label(self.inputwindow, text = " " ) 
        self.particles.pack()
  
        # Create button, it will change label text
        par_button = Button(self.inputwindow, text = "Update", command = partial(self.show, self.particles, par_clicked)).pack()
  

        ### POTENTIAL
  
        # Dropdown menu options 
        pot_options = [
          "Harmonic Oscillator", 
          "No potential", 
        ] 
        # datatype of menu text 
        pot_clicked = StringVar() 
  
        # initial menu text 
        pot_clicked.set("Harmonic Oscillator") 
  
        # Create Dropdown menu 
        pot_drop = OptionMenu(self.inputwindow, pot_clicked, *pot_options ) 
        pot_drop.pack() 

        # Create Label 
        self.potential = Label(self.inputwindow, text = " " ) 
        self.potential.pack()
  
        # Create button, it will change label text 
        pot_button = Button(self.inputwindow, text = "Update", command = partial(self.show, self.potential, pot_clicked)).pack()

        ### TIME_PARAMETERS

        time_label = Label(self.inputwindow, text = "Enter total time to run the simulation, in femtoseconds (example: 2)")
        time_label.pack()
        self.time_slider = Scale(self.inputwindow, from_=0.1, to=10, resolution=0.1, orient = HORIZONTAL)
        self.time_slider.pack()
            
        steps_label = Label(self.inputwindow, text = "Enter amount of timesteps to carry out (example: 1000)")
        steps_label.pack()
        self.timesteps_slider = Scale(self.inputwindow, from_=100, to=10000, resolution = 100, orient = HORIZONTAL)
        self.timesteps_slider.pack()
        
        osc_time_label = Label(self.inputwindow, text = "Enter a time period for the harmonic oscillator (example: 1)")
        osc_time_label.pack()
        self.osc_time_slider = Scale(self.inputwindow, from_=0.0, to=5, resolution=0.1, orient = HORIZONTAL)
        self.osc_time_slider.pack()

        osc_amp_label= Label(self.inputwindow, text = "Enter amplitude of harmonic oscillator in angstroms (example: 50)")
        osc_amp_label.pack()
        self.osc_amp_slider = Scale(self.inputwindow, from_=1, to=80, orient = HORIZONTAL)
        self.osc_amp_slider.pack()


        create_button = Button(self.inputwindow,  
                     command = partial(self.create_redraw, self.inputwindow),
                     height = 2,  
                     width = 15, 
                     text = "Create Sim button") 
        create_button.pack(side = BOTTOM)
        self.inputwindow.attributes('-topmost',True)
        self.inputwindow.mainloop()




        
    def show(self, labeltoupdate, gettertouse): 
        labeltoupdate.config( text = gettertouse.get() ) 
    
        
    def create_redraw(self, window):
        if self.particles["text"] != "1":
            self.gridpoints["text"] = "128"
            print("Two particle simulation only available with 128 gridpoints.")
        redraw_button = Button(window,  
                     command = partial(self.update_animation, self.gridpoints["text"], self.potential["text"], self.osc_time_slider.get(), self.osc_amp_slider.get(), self.time_slider.get(), self.particles["text"], 20, self.timesteps_slider.get()), 
                     height = 2,  
                     width = 10, 
                     text = "Execute") 
        redraw_button.pack(side = BOTTOM)

    def update_animation(self, gridlabel, potentiallabel, hmtimeperiod, hmamplitude, totaltimelabel, particlesystemtypelabel, spatialextentlabel, timestepstodolabel):
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy() 
            #for widget in self.toolbar.winfo_children():
            #    widget.destroy()
        self.inputwindow.destroy()
        #empty frames
        for widget in self.leftframe.winfo_children():
            widget.destroy()
        for widget in self.rightframe.winfo_children():
            widget.destroy()
        for widget in self.topframe.winfo_children():
            widget.destroy()
        if particlesystemtypelabel == "1":
            particlesystemtype = singleparticle
        else:
            particlesystemtype = twoparticles
            
        if potentiallabel == "Harmonic Oscillator":
            potential = True
        else:
            potential = False

        print("Creating particle system...")
        sys = particlesystemtype()
        print("Done")

        print("Creating Hamiltonian...")
        if potential:
            H = Hamiltonian(sys, harmonicoscillator, float(hmtimeperiod), int(hmamplitude),  int(gridlabel), int(spatialextentlabel)*Å, 2)

        elif not potential:
            H = Hamiltonian(sys, harmonicoscillator, 0, 0,  int(gridlabel), int(spatialextentlabel)*Å, 2)
        print("Done")

        print("Formulating initial wavefunction...")
        if type(sys) == singleparticle:
            wfc = GPE([H.N, H.N], H)
            psi = wfc.initialsinglewavefunction(H)
        elif type(sys) == twoparticles:
            wfc = GPE([H.N, H.N, H.N, H.N], H)
            psi = wfc.initialdualwavefunction(H)
        print("Done")

        print("Beginning time evolution...")
        total_time = float(totaltimelabel) * femtoseconds
        simulationone = Simulation(H, float(totaltimelabel)*femtoseconds, int(timestepstodolabel))

        evolvedwfcs = simulationone.sim(psi, H)
        print("Done")
        self.simulation = simulationone
        self.H = simulationone.H # creates Hamiltonian attribute
        self.animate((10, 10), 10, 30, 0.8, 0.8)
        self.Buttons()
        self.DisplayParameters()
        self.InputDialogue()


    def Buttons(self):
        # Buttons
        restart_button = Button(self.leftframe,
                            command = self.ToggleRestartStatus, 
                            height = 2,  
                            width = 10, 
                            text = "Restart") 
        restart_button.pack(side = TOP)

        pause_button = Button(self.leftframe,  
                            command = self.TogglePauseStatus, 
                            height = 2,  
                            width = 10, 
                            text = "Pause") 
        pause_button.pack(side = TOP)

        download_button = Button(self.leftframe,  
                          command = self.mpeg_write, 
                          height = 2,  
                          width = 10, 
                          text = "Download MP4") 
        download_button.pack()
        return

    def DisplayParameters(self):
        # parameter labelling

        N_lbl=Label(self.rightframe, text="Gridpoints: " + str(self.simulation.H.N))
        N_lbl.pack(side = TOP)
        
        if self.simulation.H.particle_system.m == 1:
            masstype = "Electron Mass"
        else:
            masstype = "Helium Boson Mass"

        parsys_lbl=Label(self.rightframe, text="Particle system type: " + str(self.simulation.H.particle_system.__class__.__name__) + "     Mass: " + masstype)
        parsys_lbl.pack(side = TOP)
        
        pot_lbl=Label(self.rightframe, text="Potential type: " + str(self.simulation.H.potential.__name__))
        pot_lbl.pack()
        
        if self.simulation.H.potential == "harmonicoscillator":
            pottmp_lbl=Label(self.rightframe, text="Time Period of Harmonic Oscillator: " + str(self.simulation.H.pot_time_period))
            pottmp_lbl.pack()
            potamp_lbl=Label(self.rightframe, text="Maximum Amplitude of Harmonic Oscillator: " + str(self.simulation.H.pot_max_amp))
            potamp_lbl.pack()

        sim_time_lbl=Label(self.rightframe, text="Simulation Time: " + str(self.simulation.total_time/femtoseconds)+ " fms")
        sim_time_lbl.pack()

        time_steps_lbl=Label(self.rightframe, text="Number of timesteps: " + str(self.simulation.store_steps))
        time_steps_lbl.pack(side = BOTTOM)
        return
        
    def MainWindow(self):
        # GUI
        # visualisation
        # create root window
        self.root = Tk()
 
        # root window title and dimension
        self.root.title("Simulation")
        #Set geometry (widthxheight)
        self.root.geometry('1920x1080')

        self.frame = Frame(self.root)
        self.frame.pack()

        self.leftframe = Frame(self.root)
        self.leftframe.pack(side = LEFT)

        self.rightframe = Frame(self.root)
        self.rightframe.pack(side = RIGHT)
        
        self.topframe = Frame(self.root)
        self.topframe.pack(side = TOP)

        self.InputDialogue()
        self.root.mainloop()
        

# Main program

print("SSFT Particle Simulator - Enter the desired parameters to run a simulation.")
print("Testing Mode: ") # White-Box Testing statement
# Instantiate visualisation object
graph = visualise_wavefunction()
# Create simulation window
graph.MainWindow()

# Main Program END