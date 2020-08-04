import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

ionized_impurity_scattering = 1
acoustic_deformation_potential_scattering = 1
optical_deformation_potential_scattering = 1
intervalley_deformation_potential_scattering = 1
polar_optical_scattering = 1
plot_stuff = 0

e = 1.602e-19 #charge of electron in eV
V = 1 #volume? volume of what? a unit cell? Make it one cubic centimeter?
hbar_ev = 6.58212e-16 #reduced planck constant in eV*s
hbar_j = 1.0546e-34
Temp = 300 #in kelvin
k_B = 1.38e-23
m_e = 9.109384e-31

energy_min = 1e-10 #in ev
energy_max = 2 # in eV
numberOfEnergies = 20000
energies = np.linspace(energy_min,energy_max,numberOfEnergies)*e
numberOfAngles = 100
angles = np.linspace(0,np.pi,numberOfAngles)

def phononOccupancy(energy):
    return (np.exp(energy/(k_B*Temp)) - 1)**(-1)

def DensityOfStates(energy,mass):
    if np.min(energy) < 0:
        print("error: negative energy!")
        quit
    return np.sqrt(2)*mass**1.5/(np.pi**2*hbar_j**3)*np.sqrt(energy)


n_i = 1e17 #intrinsic carrier concentration in GaAs in cm^-3
n_i *= 1e6 #convert to m^-3
Z = 1
density = 5.4e3
eps_low = 12.9
eps_high = 10.9
eps = 12.9
epsilon_0 = 8.8541878128e-12 #in farads per meter
piezo_constant = 0.16
# screeninglength_meters = 2250e-6 #obtained from https://www.keysight.com/upload/cmc_upload/All/EE_REF_PROPERTIES_Si_Ga.pdf
screeninglength_meters = np.sqrt(eps*epsilon_0*k_B*Temp/e**2/n_i)
u_l = 5.24e3 #speed of sound in gallium arsenide in m/s
c_l = density*u_l**2
alpha_gamma = 0.610
alpha_L = 0.461
alpha_X = 0.204
acoustic_deformation_potential_gamma = 7.01*1.602e-19
acoustic_deformation_potential_L = 9.2*1.602e-19
acoustic_deformation_potential_X = 9.0*1.602e-19
optical_deformation_potential = 3e8*1.602e-19*1e2 #optical deformation potential for L valley in joules per meter
optical_long_freq_gamma = 8.76e12
optical_long_freq_L = 8.76e12
optical_long_freq_X = 8.76e12
optical_energy_gamma = optical_long_freq_gamma*hbar_j
optical_energy_L = optical_long_freq_L*hbar_j
optical_energy_X = optical_long_freq_X*hbar_j
optical_energy_index_gamma = np.argmin(np.abs(energies-optical_energy_gamma))
optical_energy_index_L = np.argmin(np.abs(energies-optical_energy_L))
optical_energy_index_X = np.argmin(np.abs(energies-optical_energy_X))
mass = 0.063*m_e #from Lundstrom, definitely worth fact checking
mass_gamma = 0.063*m_e
mass_L_longitudinal = 1.9*m_e
mass_L_transverse = 0.075*m_e
mass_X_longitudinal = 1.9*m_e
mass_X_transverse = 0.19*m_e
mass_L = (mass_L_transverse**2*mass_L_longitudinal)**(1/3)
mass_X = (mass_X_transverse**2*mass_X_longitudinal)**(1/3)
intervalley_deformation_potential_gammaL = 10e10*e #in J per meter
intervalley_deformation_potential_gammaX = 10e10*e #in J per meter
intervalley_deformation_potential_LL = 10e10*e #in J per meter
intervalley_deformation_potential_LX = 5e10*e
intervalley_deformation_potential_XX = 7e10*e
intervalley_phonon_energy_gammaL = 0.0278*e #in J
intervalley_phonon_energy_gammaX = 0.0299*e #in J
intervalley_phonon_energy_LL = 0.029*e
intervalley_phonon_energy_LX = 0.0293*e
intervalley_phonon_energy_XX = 0.0299*e
energy_separation_gammaL = 0.29*e
energy_separation_gammaX = 0.48*e
energy_separation_LX = energy_separation_gammaX - energy_separation_gammaL
### some of these constants were obtained from http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/bandstr.html ###
### some were obtained from Sze's textbook ###


##### Impact ionization rate calculation here #####


def IonizedImpurityScatteringRate(energy,alpha,mass):
    C = n_i*e**4/(32*np.sqrt(2*mass)*np.pi*(eps*epsilon_0)**2)
    gamma_squared = 8*mass*energy*screeninglength_meters**2/hbar_j**2
    return C*gamma_squared**2/(1+gamma_squared)*energy**(-1.5)

def AcousticDeformationElasticScatteringRate(energy,acoustic_deformation_potential,mass):
    # return np.sqrt(2*mass**3)*(acoustic_deformation_potential)**2*k_B*Temp/(np.pi*hbar_j**4*c_l)*np.sqrt(energy)
    return np.pi*acoustic_deformation_potential**2*k_B*Temp/(hbar_j*c_l)*DensityOfStates(energy,mass)

def OpticalDeformationAbsorptionScatteringRate(energy,optical_long_freq,mass):
    return np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*phononOccupancy(optical_long_freq*hbar_j)*DensityOfStates(energy+optical_long_freq*hbar_j,mass)


def OpticalDeformationEmissionScatteringRate(energy,optical_long_freq,mass):
    output = np.zeros_like(energy)
    output[energy > optical_long_freq*hbar_j] = np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*(phononOccupancy(optical_long_freq*hbar_j)+1)*DensityOfStates(energy[energy > optical_long_freq*hbar_j]-optical_long_freq*hbar_j,mass)
    return output

def IntervalleyAbsorptionScatteringRate(energy,mass,D_ij,E_ij,energy_separation=0):
    output = np.zeros_like(energy)
    if energy_separation < 0:
        output = np.pi*D_ij**2/(2*density*E_ij/hbar_j)*phononOccupancy(E_ij)*DensityOfStates(energy+E_ij-energy_separation,mass)
    else:
        output[energy + E_ij > energy_separation] = np.pi*D_ij**2/(2*density*E_ij/hbar_j)*phononOccupancy(E_ij)*DensityOfStates(energy[energy + E_ij > energy_separation]+E_ij-energy_separation,mass)
    return output

def IntervalleyEmissionScatteringRate(energy,mass,D_ij,E_ij,energy_separation=0):
    output = np.zeros_like(energy)
    if energy_separation < 0:
        output[energy > E_ij] = np.pi*D_ij**2/(2*density*E_ij/hbar_j)*(phononOccupancy(E_ij)+1)*DensityOfStates(energy[energy > E_ij]-E_ij-energy_separation,mass)
    if energy_separation >= 0:
        output[energy > E_ij + energy_separation] = np.pi*D_ij**2/(2*density*E_ij/hbar_j)*(phononOccupancy(E_ij)+1)*DensityOfStates(energy[energy > E_ij + energy_separation]-E_ij-energy_separation,mass)
    return output

def PolarOpticalAbsorptionScatteringRate(energy,optical_long_freq,mass):
    optical_energy = optical_long_freq*hbar_j
    A = e**2*optical_long_freq*(eps_low/eps_high-1)/(2*np.pi*eps_low*epsilon_0*hbar_j*np.sqrt(2*energy/mass))
    # A = e**2*np.sqrt(mass)*optical_long_freq/(2*np.sqrt(2)*np.pi*hbar_j**2)*(1/eps_high - 1/eps_low)*energy**(-0.5)
    N_0 = phononOccupancy(optical_energy)
    B = N_0*np.arcsinh(np.sqrt(energy/optical_energy))
    return A*B

def PolarOpticalEmissionScatteringRate(energy,optical_long_freq,mass):
    optical_energy = optical_long_freq*hbar_j
    A = e**2*optical_long_freq*(eps_low/eps_high-1)/(2*np.pi*eps_low*epsilon_0*hbar_j*np.sqrt(2*energy/mass))
    # A = e**2*np.sqrt(mass)*optical_long_freq/(2*np.sqrt(2)*np.pi*hbar_j**2)*(1/eps_high - 1/eps_low)*energy**(-0.5)
    N_0 = phononOccupancy(optical_energy)
    B = np.zeros_like(energy)
    B[energy > optical_energy] = (N_0+1)*np.arcsinh(np.sqrt(energy[energy > optical_energy]/optical_energy-1))
    return A*B


scatIndex = 1

if ionized_impurity_scattering == 1:
    ionized_rate_gamma = IonizedImpurityScatteringRate(energies,alpha_gamma,mass_gamma)
    ionized_rate_L_long = IonizedImpurityScatteringRate(energies,alpha_L,mass_L_longitudinal)
    ionized_rate_L_tran = IonizedImpurityScatteringRate(energies,alpha_L,mass_L_transverse)
    ionized_rate_X_long = IonizedImpurityScatteringRate(energies,alpha_X,mass_X_longitudinal)
    ionized_rate_X_tran = IonizedImpurityScatteringRate(energies,alpha_X,mass_X_transverse)
    if scatIndex == 1:
        scatteringRateTable = np.array([ionized_rate_gamma])
        scatTypeLabels = {scatIndex: 'Ionized Impurity, Gamma Valley'}
        scatTypeEnergyShifts = {scatIndex: 0}
    else:
        scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_gamma]),axis=0)
        scatTypeLabels[scatIndex] = 'Ionized Impurity, Gamma Valley'
        scatTypeEnergyShifts[scatIndex] = 0
    ii1 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_L_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, L Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = 0
    ii2 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_L_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, L Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    ii3 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_X_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, X Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = 0
    ii4 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_X_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, X Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    ii5 = scatIndex - 1
    scatIndex += 1

if acoustic_deformation_potential_scattering == 1:
    acoustic_rates_gamma = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_gamma,mass_gamma) #double it to take absorption + emission into account
    acoustic_rates_L_long = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_L,mass_L_longitudinal)
    acoustic_rates_L_tran = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_L,mass_L_transverse)
    acoustic_rates_X_long = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_X,mass_X_longitudinal)
    acoustic_rates_X_tran = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_X,mass_X_transverse)
    if scatIndex == 1:
        scatteringRateTable = np.array([acoustic_rates_gamma])
        scatTypeLabels = {scatIndex: 'Acoustic Deformation Potential, Gamma Valley'}
        scatTypeEnergyShifts = {scatIndex: 0}
    else:
        scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_gamma]),axis=0)
        scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, Gamma Valley'
        scatTypeEnergyShifts[scatIndex] = 0
        adp1 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_L_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, L Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = 0
    adp2 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_L_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, L Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    adp3 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_X_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, X Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = 0
    adp4 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_X_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, X Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    adp5 = scatIndex - 1
    scatIndex += 1

if optical_deformation_potential_scattering == 1:
    optical_absorption_rates_gamma = OpticalDeformationAbsorptionScatteringRate(energies,optical_long_freq_gamma,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy_gamma
    odp1 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_gamma = OpticalDeformationEmissionScatteringRate(energies,optical_long_freq_gamma,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_gamma
    odp2 = scatIndex - 1
    scatIndex += 1

    optical_absorption_rates_L = OpticalDeformationAbsorptionScatteringRate(energies,optical_long_freq_L,(mass_L_transverse**2*mass_L_longitudinal)**(1/3))
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, L Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy_L
    odp3 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_L = OpticalDeformationEmissionScatteringRate(energies,optical_long_freq_L,(mass_L_transverse**2*mass_L_longitudinal)**(1/3))
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, L Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_L
    odp4 = scatIndex - 1
    scatIndex += 1

    optical_absorption_rates_X = OpticalDeformationAbsorptionScatteringRate(energies,optical_long_freq_X,(mass_X_transverse**2*mass_X_longitudinal)**(1/3))
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, X Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy_X
    odp5 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_X = OpticalDeformationEmissionScatteringRate(energies,optical_long_freq_X,(mass_X_transverse**2*mass_X_longitudinal)**(1/3))
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, X Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_X
    odp6 = scatIndex - 1
    scatIndex += 1

if intervalley_deformation_potential_scattering == 1:

    rate_gammaL_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL-energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL
    ivdp1 = scatIndex - 1
    scatIndex += 1

    rate_gammaL_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL-energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL
    ivdp2 = scatIndex - 1
    scatIndex += 1

    rate_Lgamma_abs = IntervalleyAbsorptionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=-energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Lgamma_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to Gamma'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL+energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL
    ivdp3 = scatIndex - 1
    scatIndex += 1


    rate_gammaX_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX-energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX
    ivdp4 = scatIndex - 1
    scatIndex += 1

    rate_gammaX_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX-energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX
    ivdp5 = scatIndex - 1
    scatIndex += 1

    rate_Xgamma_abs = IntervalleyAbsorptionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=-energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Xgamma_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to Gamma'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX+energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX
    ivdp6 = scatIndex - 1
    scatIndex += 1


    rate_LX_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX
    ivdp7 = scatIndex - 1
    scatIndex += 1

    rate_LX_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX
    ivdp8 = scatIndex - 1
    scatIndex += 1

    rate_XL_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX+energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX
    ivdp9 = scatIndex - 1
    scatIndex += 1

    rate_XL_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX+energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX
    ivdp10 = scatIndex - 1
    scatIndex += 1


    rate_LL_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LL
    ivdp11 = scatIndex - 1
    scatIndex += 1

    rate_LL_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LL
    ivdp12 = scatIndex - 1
    scatIndex += 1


    rate_XX_abs_long = IntervalleyAbsorptionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_XX
    ivdp13 = scatIndex - 1
    scatIndex += 1

    rate_XX_abs_tran = IntervalleyAbsorptionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_XX
    ivdp14 = scatIndex - 1
    scatIndex += 1


    rate_gammaL_emi_long = IntervalleyEmissionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL-energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL
    ivdp15 = scatIndex - 1
    scatIndex += 1

    rate_gammaL_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL-energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL
    ivdp16 = scatIndex - 1
    scatIndex += 1

    rate_Lgamma_emi = IntervalleyEmissionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=-energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Lgamma_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to Gamma'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL+energy_separation_gammaL
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL
    ivdp17 = scatIndex - 1
    scatIndex += 1


    rate_gammaX_emi_long = IntervalleyEmissionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX-energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX
    ivdp18 = scatIndex - 1
    scatIndex += 1

    rate_gammaX_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX-energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX
    ivdp19 = scatIndex - 1
    scatIndex += 1

    rate_Xgamma_emi = IntervalleyEmissionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=-energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Xgamma_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to Gamma'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX+energy_separation_gammaX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX
    ivdp20 = scatIndex - 1
    scatIndex += 1

    rate_LX_emi_long = IntervalleyEmissionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX
    ivdp21 = scatIndex - 1
    scatIndex += 1

    rate_LX_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX
    ivdp22 = scatIndex - 1
    scatIndex += 1

    rate_XL_emi_long = IntervalleyEmissionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX+energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX
    ivdp23 = scatIndex - 1
    scatIndex += 1

    rate_XL_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX+energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX
    ivdp24 = scatIndex - 1
    scatIndex += 1


    rate_LL_emi_long = IntervalleyEmissionScatteringRate(energies,mass_L_longitudinal,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to L, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LL
    ivdp25 = scatIndex - 1
    scatIndex += 1

    rate_LL_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_L_transverse,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to L, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LL
    ivdp26 = scatIndex - 1
    scatIndex += 1


    rate_XX_emi_long = IntervalleyEmissionScatteringRate(energies,mass_X_longitudinal,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_emi_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to X, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_XX
    ivdp27 = scatIndex - 1
    scatIndex += 1

    rate_XX_emi_tran = IntervalleyEmissionScatteringRate(energies,mass_X_transverse,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_emi_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to X, Transverse'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_XX
    ivdp28 = scatIndex - 1
    scatIndex += 1

if polar_optical_scattering == 1:

    polar_optical_absorption_rates_gamma = PolarOpticalAbsorptionScatteringRate(energies,optical_long_freq_gamma,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy_gamma
    pop1 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_gamma = PolarOpticalEmissionScatteringRate(energies,optical_long_freq_gamma,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_gamma
    pop2 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_L_long = PolarOpticalAbsorptionScatteringRate(energies,optical_long_freq_L,mass_L_longitudinal)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_L_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, L Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = optical_energy_L
    pop3 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_L_tran = PolarOpticalAbsorptionScatteringRate(energies,optical_long_freq_L,mass_L_transverse)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_L_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, L Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = optical_energy_L
    pop4 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_L_long = PolarOpticalEmissionScatteringRate(energies,optical_long_freq_L,mass_L_longitudinal)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_L_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, L Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_L
    pop5 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_L_tran = PolarOpticalEmissionScatteringRate(energies,optical_long_freq_L,mass_L_transverse)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_L_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, L Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_L
    pop6 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_X_long = PolarOpticalAbsorptionScatteringRate(energies,optical_long_freq_X,mass_X_longitudinal)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_X_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, X Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = optical_energy_X
    pop7 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_X_tran = PolarOpticalAbsorptionScatteringRate(energies,optical_long_freq_X,mass_X_transverse)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_X_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, X Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = optical_energy_X
    pop8 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_X_long = PolarOpticalEmissionScatteringRate(energies,optical_long_freq_X,mass_X_longitudinal)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_X_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, X Valley, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_X
    pop9 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_X_tran = PolarOpticalEmissionScatteringRate(energies,optical_long_freq_X,mass_X_transverse)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_X_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, X Valley, Transverse'
    scatTypeEnergyShifts[scatIndex] = -optical_energy_X
    pop10 = scatIndex - 1
    scatIndex += 1
quit
if plot_stuff == 1:
    plt.figure(0)
    plt.title('Ionized Impurity')
    plt.plot(energies/e,scatteringRateTable[ii1],label='gamma')
    plt.plot(energies/e,scatteringRateTable[ii2],label='L, long')
    plt.plot(energies/e,scatteringRateTable[ii3],label="L, tran")
    plt.plot(energies/e,scatteringRateTable[ii4],label="X, long")
    plt.plot(energies/e,scatteringRateTable[ii5],label="X, tran")
    plt.legend()
    plt.figure(1)
    plt.title('acoustic deformation')
    plt.plot(energies/e,scatteringRateTable[adp1],label="gamma")
    plt.plot(energies/e,scatteringRateTable[adp2],label='L, long')
    plt.plot(energies/e,scatteringRateTable[adp3],label="L, tran")
    plt.plot(energies/e,scatteringRateTable[adp4],label="X, long")
    plt.plot(energies/e,scatteringRateTable[adp5],label="X, tran")
    plt.legend()
    plt.figure(2)
    plt.title('optical deformation')
    plt.plot(energies/e,scatteringRateTable[odp1],label='gamma, absorption')
    plt.plot(energies/e,scatteringRateTable[odp2],label='gamma, emission')
    plt.plot(energies/e,scatteringRateTable[odp3],label='L, absorption')
    plt.plot(energies/e,scatteringRateTable[odp4],label='L, emission')
    plt.plot(energies/e,scatteringRateTable[odp5],label='X, absorption')
    plt.plot(energies/e,scatteringRateTable[odp6],label='X, emission')
    plt.legend()
    plt.figure(3)
    plt.title('intervalley, from gamma valley')
    plt.plot(energies/e,scatteringRateTable[ivdp1],label='absorption, gamma to L, long')
    plt.plot(energies/e,scatteringRateTable[ivdp2],label='absorption, gamma to L, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp15],label='emission, gamma to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp16],label='emission, gamma to X, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp5],label='absorption, gamma to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp6],label='absorption, gamma to X, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp18],label='emission, gamma to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp19],label='emission, gamma to X, tran')
    plt.legend()
    plt.figure(4)
    plt.title('intervalley, from L valley')
    plt.plot(energies/e,scatteringRateTable[ivdp3],label='absorption, L to gamma')
    plt.plot(energies/e,scatteringRateTable[ivdp7],label='absorption, L to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp8],label='absorption, L to X, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp11],label='absorption, L to L, long')
    plt.plot(energies/e,scatteringRateTable[ivdp12],label='absorption, L to L, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp17],label='emission, L to gamma')
    plt.plot(energies/e,scatteringRateTable[ivdp21],label='emission, L to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp22],label='emission, L to X, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp25],label='emission, L to L, long')
    plt.plot(energies/e,scatteringRateTable[ivdp26],label='emission, L to L, tran')
    plt.legend()
    plt.figure(5)
    plt.title('intervalley, from X valley')
    plt.plot(energies/e,scatteringRateTable[ivdp6],label='absorption, X to gamma')
    plt.plot(energies/e,scatteringRateTable[ivdp9],label='absorption, X to L, long')
    plt.plot(energies/e,scatteringRateTable[ivdp10],label='absorption, X to L, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp13],label='absorption, X to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp14],label='absorption, X to X, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp20],label='emission, X to gamma')
    plt.plot(energies/e,scatteringRateTable[ivdp23],label='emission, X to L, long')
    plt.plot(energies/e,scatteringRateTable[ivdp24],label='emission, X to L, tran')
    plt.plot(energies/e,scatteringRateTable[ivdp27],label='emission, X to X, long')
    plt.plot(energies/e,scatteringRateTable[ivdp28],label='emission, X to X, tran')
    plt.legend()
    plt.figure(6)
    plt.title('polar optical, gamma')
    plt.plot(energies/e,scatteringRateTable[pop1],label='absorption, gamma')
    plt.plot(energies/e,scatteringRateTable[pop2],label='emission, gamma')
    plt.yscale('log')
    plt.figure(7)
    plt.title('polar optical, L')
    plt.plot(energies/e,scatteringRateTable[pop3],label='absorption, L, long')
    plt.plot(energies/e,scatteringRateTable[pop4],label='absorption, L, tran')
    plt.plot(energies/e,scatteringRateTable[pop5],label='emission, L, long')
    plt.plot(energies/e,scatteringRateTable[pop6],label='emission, L, long')
    plt.legend()
    plt.yscale('log')
    plt.figure(8)
    plt.title('polar optical, X')
    plt.plot(energies/e,scatteringRateTable[pop7],label='absorption, X, long')
    plt.plot(energies/e,scatteringRateTable[pop8],label='absorption, X, tran')
    plt.plot(energies/e,scatteringRateTable[pop9],label='emission, X, long')
    plt.plot(energies/e,scatteringRateTable[pop10],label='emission, X, long')
    plt.legend()
    plt.yscale('log')
    plt.show()
    quit

scatteringSums = np.zeros_like(scatteringRateTable)
for i in range(len(scatteringRateTable)):
    if i == 0:
        scatteringSums[i] = scatteringRateTable[i]
    if i > 0:
        scatteringSums[i] = scatteringRateTable[i]+scatteringSums[i-1]
max_scatter = np.max(scatteringSums[-1])

scatTypeConverterGamma = np.array([ii1,adp1,odp1,ivdp1,ivdp2,ivdp4,ivdp5,ivdp15,ivdp16,ivdp18,ivdp19,pop1,pop2])
scatteringRateTableGamma = scatteringRateTable[scatTypeConverterGamma]
scatteringRateTableGamma[4] *= 3 #3 equivalent L transverse modes (1 long, 3 trans is half of 2 long, 6 trans as discussed with Steve)
scatteringRateTable[6] *= 2 #2 equivalent X transverse valleys
scatteringRateTableGamma[8] *= 3
scatteringRateTableGamma[10] *= 2
scatteringSumsGamma = np.zeros_like(scatteringRateTableGamma)
for i in range(len(scatteringRateTableGamma)):
    if i == 0:
        scatteringSumsGamma[i] = scatteringRateTableGamma[i]
    if i > 0:
        scatteringSumsGamma[i] = scatteringRateTableGamma[i]+scatteringSumsGamma[i-1]
max_scatter_gamma = np.max(scatteringSumsGamma[-1])

scatTypeConverterL_longitudinal = np.array([ii2,adp2,odp3,odp4,ivdp3,ivdp7,ivdp8,ivdp11,ivdp12,ivdp17,ivdp21,ivdp22,ivdp25,ivdp26,pop3,pop5])
scatteringRateTableL_longitudinal = scatteringRateTable[scatTypeConverterL_longitudinal]
scatteringRateTableL_longitudinal[6] *= 2 #2 transverse modes in X valley
scatteringRateTableL_longitudinal[7] *= 0 #since there is only 1 longitudinal mode in L valley and it is being scattered from, there are 0 available longitudinal modes
scatteringRateTableL_longitudinal[8] *= 3 #3 total L transverse modes in one brillouin zone
scatteringRateTableL_longitudinal[11] *= 2 #2 transverse modes in X valley
scatteringRateTableL_longitudinal[12] *= 0
scatteringRateTableL_longitudinal[13] *= 3
scatteringSumsL_longitudinal = np.zeros_like(scatteringRateTableL_longitudinal)
for i in range(len(scatteringRateTableL_longitudinal)):
    if i == 0:
        scatteringSumsL_longitudinal[i] = scatteringRateTableL_longitudinal[i]
    if i > 0:
        scatteringSumsL_longitudinal[i] = scatteringRateTableL_longitudinal[i]+scatteringSumsL_longitudinal[i-1]
max_scatter_L_longitudinal = np.max(scatteringSumsL_longitudinal[-1])

scatTypeConverterL_transverse = np.array([ii2,adp2,odp3,odp4,ivdp3,ivdp7,ivdp8,ivdp11,ivdp12,ivdp17,ivdp21,ivdp22,ivdp25,ivdp26,pop4,pop6])
scatteringRateTableL_transverse = scatteringRateTable[scatTypeConverterL_transverse]
scatteringRateTableL_transverse[6] *= 2
scatteringRateTableL_transverse[8] *= 2
scatteringRateTableL_transverse[11] *= 2
scatteringRateTableL_transverse[13] *= 2
scatteringSumsL_transverse = np.zeros_like(scatteringRateTableL_transverse)
for i in range(len(scatteringRateTableL_transverse)):
    if i == 0:
        scatteringSumsL_transverse[i] = scatteringRateTableL_transverse[i]
    if i > 0:
        scatteringSumsL_transverse[i] = scatteringRateTableL_transverse[i]+scatteringSumsL_transverse[i-1]
max_scatter_L_transverse = np.max(scatteringSumsL_transverse[-1])

scatTypeConverterX_longitudinal = np.array([ii4,adp4,odp5,odp6,ivdp6,ivdp9,ivdp10,ivdp13,ivdp14,ivdp20,ivdp23,ivdp24,ivdp27,ivdp28,pop7,pop9])
scatteringRateTableX_longitudinal = scatteringRateTable[scatTypeConverterX_longitudinal]
scatteringRateTableX_longitudinal[6] *= 3 #three transverse modes available in L valley
scatteringRateTableX_longitudinal[7] *= 0 #no available longitudinal modes in X valley
scatteringRateTableX_longitudinal[8] *= 2 #two available trans modes in X valley
scatteringRateTableX_longitudinal[11] *= 3
scatteringRateTableX_longitudinal[12] *= 0
scatteringRateTableX_longitudinal[13] *= 2
scatteringSumsX_longitudinal = np.zeros_like(scatteringRateTableX_longitudinal)
for i in range(len(scatteringRateTableX_longitudinal)):
    if i == 0:
        scatteringSumsX_longitudinal[i] = scatteringRateTableX_longitudinal[i]
    if i > 0:
        scatteringSumsX_longitudinal[i] = scatteringRateTableX_longitudinal[i]+scatteringSumsX_longitudinal[i-1]
max_scatter_X_longitudinal = np.max(scatteringSumsX_longitudinal[-1])

scatTypeConverterX_transverse = np.array([ii4,adp4,odp5,odp6,ivdp6,ivdp9,ivdp10,ivdp13,ivdp14,ivdp20,ivdp23,ivdp24,ivdp27,ivdp28,pop8,pop10])
scatteringRateTableX_transverse = scatteringRateTable[scatTypeConverterX_transverse]
scatteringRateTableX_transverse[6] *= 3
scatteringRateTableX_transverse[11] *= 3
scatteringSumsX_transverse = np.zeros_like(scatteringRateTableX_transverse)
for i in range(len(scatteringRateTableX_transverse)):
    if i == 0:
        scatteringSumsX_transverse[i] = scatteringRateTableX_transverse[i]
    if i > 0:
        scatteringSumsX_transverse[i] = scatteringRateTableX_transverse[i]+scatteringSumsX_transverse[i-1]
max_scatter_X_transverse = np.max(scatteringSumsX_transverse[-1])


def energizer(vx,vy,vz,mass):
    """Returns energy (in Joules) of a particle with velocity components vx,vy,vz."""
    return 0.5*mass*(vx**2+vy**2+vz**2)

def rotateVector(vx,vy,vz,theta,phi):
    if type(theta) == np.ndarray:
        vx_rotated = np.zeros_like(vx)
        vy_rotated = np.zeros_like(vy)
        vz_rotated = np.zeros_like(vz)
        for i in range(len(theta)):
            vMagnitude = np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2)
            if vx[i] != 0:
                framePhi = np.arctan(vy[i]/vx[i])
            else:
                framePhi = 0
            if vz[i] != 0:
                frameTheta = np.arctan(np.sqrt(vx[i]**2+vy[i]**2)/np.abs(vz[i]))
            else:
                frameTheta = 0
            vx_temp = vMagnitude*np.sin(theta[i])*np.cos(phi[i])
            vy_temp = vMagnitude*np.sin(theta[i])*np.sin(phi[i])
            vz_temp = vMagnitude*np.cos(theta[i])
            vx_rotated[i] = np.cos(framePhi)*np.cos(frameTheta)*vx_temp + np.sin(framePhi)*vy_temp - np.cos(framePhi)*np.sin(frameTheta)*vz_temp
            vy_rotated[i] = -np.sin(framePhi)*np.cos(frameTheta)*vx_temp + np.cos(framePhi)*vy_temp + np.sin(framePhi)*np.sin(frameTheta)*vz_temp
            vz_rotated[i] = np.sin(frameTheta)*vx_temp + np.cos(frameTheta)*vz_temp
    else:
        vMagnitude = np.sqrt(vx**2+vy**2+vz**2)
        framePhi = np.arctan(vy/vx)
        frameTheta = np.arctan(np.sqrt(vx**2+vy**2)/vz)
        vx_temp = vMagnitude*np.sin(theta)*np.cos(phi)
        vy_temp = vMagnitude*np.sin(theta)*np.sin(phi)
        vz_temp = vMagnitude*np.cos(theta)
        vx_rotated = np.cos(framePhi)*np.cos(frameTheta)*vx_temp + np.sin(framePhi)*vy_temp - np.cos(framePhi)*np.sin(frameTheta)*vz_temp
        vy_rotated = -np.sin(framePhi)*np.cos(frameTheta)*vx_temp + np.cos(framePhi)*vy_temp + np.sin(framePhi)*np.sin(frameTheta)*vz_temp
        vz_rotated = np.sin(frameTheta)*vx_temp + np.cos(frameTheta)*vz_temp

    return vx_rotated,vy_rotated,vz_rotated

def RandomThetaSelectorIonizedImpurityScattering(energy,mass):
    tryAgain = 1
    ksquared = 2*mass*energy/hbar_j**2
    cos_max = (2*ksquared)/(2*ksquared+1/screeninglength_meters**2) #not sure if this is correct, but it's what I got
    sin_max = np.sin(np.arccos(cos_max))
    max = sin_max/(0.5-0.5*cos_max+1/(4*ksquared*screeninglength_meters**2))
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < np.sin(randomTheta)/(0.5-0.5*np.cos(randomTheta)+1/(4*ksquared*screeninglength_meters**2)):
            tryAgain = 0
            return randomTheta

def polarOpticalAbsorptionRandomTheta(vx,vy,vz,mass,optical_energy):
    tryAgain = 1
    E_i = 0.5*mass*(vx**2+vy**2+vz**2)
    E_f = E_i + optical_energy
    max = 1/np.sqrt((E_i+E_f)**2-4*E_i*E_f) #not sure if this is correct, but it's what I got
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < np.sin(randomTheta)/(E_i+E_f-2*np.sqrt(E_i*E_f)*np.cos(randomTheta)):
            tryAgain = 0
            return randomTheta

def polarOpticalEmissionRandomTheta(vx,vy,vz,mass,optical_energy):
    tryAgain = 1
    E_i = 0.5*mass*(vx**2+vy**2+vz**2)
    E_f = E_i - optical_energy
    if E_f < 0:
        if E_f > -1e-23:
            E_f = 0
        else:
            print('Error: trying to select a theta for an impossible collision!')
            quit
    max = 1/np.sqrt((E_i+E_f)**2-4*E_i*E_f) #not sure if this is correct, but it's what I got
    tryAgain = 1
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < np.sin(randomTheta)/(E_i+E_f-2*np.sqrt(E_i*E_f)*np.cos(randomTheta)):
            tryAgain = 0
            return randomTheta


##### Generate random flight time before collision #####

time_step_count = 200
particle_count = 1000
start_time = 0
max_time = 5e-12 #run the simulation for 5 picoseconds
dt = (max_time-start_time)/time_step_count
times = np.linspace(start_time,max_time,time_step_count)

vxArray = np.random.rand(particle_count)
vyArray = np.random.rand(particle_count)
vzArray = np.random.rand(particle_count)

mass_array = np.array([mass_gamma,mass_L_longitudinal,mass_L_transverse,mass_X_longitudinal,mass_X_transverse])
optical_energy_array = np.array([optical_energy_gamma,optical_energy_L,optical_energy_L,optical_energy_X,optical_energy_X])
max_scatter_array = np.array([max_scatter_gamma,max_scatter_L_longitudinal,max_scatter_L_transverse,max_scatter_X_longitudinal,max_scatter_X_transverse])
electron_states = np.zeros(particle_count).astype(int)

energyAveragesGamma = np.zeros([time_step_count])
energyAveragesL_long = np.zeros([time_step_count])
energyAveragesL_tran = np.zeros([time_step_count])
energyAveragesX_long = np.zeros([time_step_count])
energyAveragesX_tran = np.zeros([time_step_count])
energyAverages = np.zeros([time_step_count])
velocityAveragesGamma = np.zeros([time_step_count])
velocityAveragesL_long = np.zeros([time_step_count])
velocityAveragesL_tran = np.zeros([time_step_count])
velocityAveragesX_long = np.zeros([time_step_count])
velocityAveragesX_tran = np.zeros([time_step_count])
velocityAverages = np.zeros([time_step_count])

electric_field = np.array([0,0,2e6])
scatCount = np.zeros(len(scatteringRateTable)).astype(int)
scatCount2 = scatCount.copy()
for stepNum in range(time_step_count):
    print(scatCount-scatCount2)
    scatCount2 = scatCount.copy()

    print('Starting on time interval ', stepNum, ' out of ', time_step_count, '.')
    timeElapsed = np.zeros(particle_count)
    randomFlightTimes = -np.log(np.random.rand(particle_count))/max_scatter_array[electron_states]
    while np.max(randomFlightTimes) > 0:
        randomFlightTimes[randomFlightTimes+timeElapsed > dt] = 0
        timeElapsed += randomFlightTimes

        vxArray = vxArray + e*electric_field[0]*randomFlightTimes/mass_array[electron_states]
        vyArray = vyArray + e*electric_field[1]*randomFlightTimes/mass_array[electron_states]
        vzArray = vzArray + e*electric_field[2]*randomFlightTimes/mass_array[electron_states]

        currentEnergies = energizer(vxArray,vyArray,vzArray,mass_array[electron_states])
        currentEnergyIndices = np.zeros([particle_count]).astype(int)
        randomScats = np.zeros(particle_count)
        randomScats[randomFlightTimes != 0] = np.random.rand(len(randomFlightTimes[randomFlightTimes != 0]))*max_scatter_array[electron_states[randomFlightTimes != 0]]
        scatYesOrNo = np.zeros(particle_count)
        scatteringType = np.zeros(particle_count).astype(int)
        scatterTheta = np.zeros(particle_count)
        scatterPhi = np.zeros(particle_count)
        for i in range(particle_count):
            currentEnergyIndex = np.argmin(np.abs(energies-currentEnergies[i]))
            if electron_states[i] == 0:
                for scatNum in range(len(scatteringSumsGamma)):
                    if randomScats[i] < scatteringSumsGamma[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        scatteringType[i] = scatTypeConverterGamma[scatNum]+1
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterGamma[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsGamma)-1:
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, Gamma Valley':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, Gamma Valley':
                        scatterTheta[i] = polarOpticalAbsorptionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, Gamma Valley':
                        scatterTheta[i] = polarOpticalEmissionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to L, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to L, Longitudinal':
                        electron_states[i] = 1
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to L, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to L, Transverse':
                        electron_states[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to X, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to X, Longitudinal':
                        electron_states[i] = 3
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to X, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to X, Transverse':
                        electron_states[i] = 4
                    energyShift = scatTypeEnergyShifts[scatteringType[i]]
                    if currentEnergies[i]+energyShift < 0:
                        if currentEnergies[i]+energyShift > -1e-23:
                            vxArray[i] = 0
                            vyArray[i] = 0
                            vzArray[i] = 0
                        else:
                            print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                            quit
                    else:
                        vxArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vyArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vzArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                    newEnergy = energizer(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]])
                    if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 5e-24:
                        print('Uh oh')
                        quit
            elif electron_states[i] == 1:
                for scatNum in range(len(scatteringSumsL_longitudinal)):
                    # print('Tried ', scatNum)
                    # print('The scattering rate compared to is ', scatteringSumsL_longitudinal[scatNum,currentEnergyIndex])
                    if randomScats[i] < scatteringSumsL_longitudinal[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        # print('Successful scattering event picked with ', scatNum)
                        scatteringType[i] = scatTypeConverterL_longitudinal[scatNum]+1
                        # print('This converted to index ', scatteringType[i])
                        # print('This corresponds to ', scatTypeLabels[scatteringType[i]])
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterL_longitudinal[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsL_longitudinal)-1:
                        # print('No luck with that scattering number')
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, L Valley, Longitudinal':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, L Valley, Longitudinal':
                        scatterTheta[i] = polarOpticalAbsorptionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, L Valley, Longitudinal':
                        scatterTheta[i] = polarOpticalEmissionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to Gamma':
                        electron_states[i] = 0
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to L, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to L, Transverse':
                        electron_states[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to X, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to X, Longitudinal':
                        electron_states[i] = 3
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to X, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to X, Transverse':
                        electron_states[i] = 4
                    energyShift = scatTypeEnergyShifts[scatteringType[i]]
                    if currentEnergies[i]+energyShift < 0:
                        if currentEnergies[i]+energyShift > -1e-23:
                            vxArray[i] = 0
                            vyArray[i] = 0
                            vzArray[i] = 0
                        else:
                            print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                            quit
                    else:
                        vxArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vyArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vzArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                    newEnergy = energizer(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]])
                    if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 5e-24:
                        print('Uh oh')
                        quit
            elif electron_states[i] == 2:
                for scatNum in range(len(scatteringSumsL_transverse)):
                    # print('Tried ', scatNum)
                    # print('The scattering rate compared to is ', scatteringSumsL_transverse[scatNum,currentEnergyIndex])
                    if randomScats[i] < scatteringSumsL_transverse[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        # print('Successful scattering event picked with ', scatNum)
                        scatteringType[i] = scatTypeConverterL_transverse[scatNum]+1
                        # print('This converted to index ', scatteringType[i])
                        # print('This corresponds to ', scatTypeLabels[scatteringType[i]])
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterL_transverse[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsL_transverse)-1:
                        # print('No luck with that scattering number')
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, L Valley, Transverse':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, L Valley, Transverse':
                        scatterTheta[i] = polarOpticalAbsorptionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, L Valley, Transverse':
                        scatterTheta[i] = polarOpticalEmissionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to Gamma':
                        electron_states[i] = 0
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to L, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to L, Longitudinal':
                        electron_states[i] = 1
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to X, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to X, Longitudinal':
                        electron_states[i] = 3
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to X, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to X, Transverse':
                        electron_states[i] = 4
                    energyShift = scatTypeEnergyShifts[scatteringType[i]]
                    if currentEnergies[i]+energyShift < 0:
                        if currentEnergies[i]+energyShift > -1e-23:
                            vxArray[i] = 0
                            vyArray[i] = 0
                            vzArray[i] = 0
                        else:
                            print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                            quit
                    else:
                        vxArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vyArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vzArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                    newEnergy = energizer(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]])
                    if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 5e-24:
                        print('Uh oh')
                        quit
            elif electron_states[i] == 3:
                for scatNum in range(len(scatteringSumsX_longitudinal)):
                    # print('Tried ', scatNum)
                    # print('The scattering rate compared to is ', scatteringSumsX_longitudinal[scatNum,currentEnergyIndex])
                    if randomScats[i] < scatteringSumsX_longitudinal[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        # print('Successful scattering event picked with ', scatNum)
                        scatteringType[i] = scatTypeConverterX_longitudinal[scatNum]+1
                        # print('This converted to index ', scatteringType[i])
                        # print('This corresponds to ', scatTypeLabels[scatteringType[i]])
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterX_longitudinal[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsX_longitudinal)-1:
                        # print('No luck with that scattering number')
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, X Valley, Longitudinal':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, X Valley, Longitudinal':
                        scatterTheta[i] = polarOpticalAbsorptionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, X Valley, Longitudinal':
                        scatterTheta[i] = polarOpticalEmissionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to Gamma':
                        electron_states[i] = 0
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to L, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to L, Longitudinal':
                        electron_states[i] = 1
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to L, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to L, Transverse':
                        electron_states[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to X, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to X, Transverse':
                        electron_states[i] = 4
                    energyShift = scatTypeEnergyShifts[scatteringType[i]]
                    if currentEnergies[i]+energyShift < 0:
                        if currentEnergies[i]+energyShift > -1e-23:
                            vxArray[i] = 0
                            vyArray[i] = 0
                            vzArray[i] = 0
                        else:
                            print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                            quit
                    else:
                        vxArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vyArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vzArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                    newEnergy = energizer(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]])
                    if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 5e-24:
                        print('Uh oh')
                        quit
            elif electron_states[i] == 4:
                for scatNum in range(len(scatteringSumsX_transverse)):
                    # print('Tried ', scatNum)
                    # print('The scattering rate compared to is ', scatteringSumsX_transverse[scatNum,currentEnergyIndex])
                    if randomScats[i] < scatteringSumsX_transverse[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        # print('Successful scattering event picked with ', scatNum)
                        scatteringType[i] = scatTypeConverterX_transverse[scatNum]+1
                        # print('This converted to index ', scatteringType[i])
                        # print('This corresponds to ', scatTypeLabels[scatteringType[i]])
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterX_transverse[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsX_transverse)-1:
                        # print('No luck with that scattering number')
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, X Valley, Transverse':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, X Valley, Transverse':
                        scatterTheta[i] = polarOpticalAbsorptionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, X Valley, Transverse':
                        scatterTheta[i] = polarOpticalEmissionRandomTheta(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]],optical_energy_array[electron_states[i]])
                        scatYesOrNo[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to Gamma':
                        electron_states[i] = 0
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to L, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to L, Longitudinal':
                        electron_states[i] = 1
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to L, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to L, Transverse':
                        electron_states[i] = 2
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to X, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to X, Longitudinal':
                        electron_states[i] = 3
                    energyShift = scatTypeEnergyShifts[scatteringType[i]]
                    if currentEnergies[i]+energyShift < 0:
                        if currentEnergies[i]+energyShift > -1e-23:
                            vxArray[i] = 0
                            vyArray[i] = 0
                            vzArray[i] = 0
                        else:
                            print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                            quit
                    else:
                        vxArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vyArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                        vzArray[i] *= np.sqrt((old_mass/mass_array[electron_states[i]])*(currentEnergies[i]+energyShift)/currentEnergies[i])
                    newEnergy = energizer(vxArray[i],vyArray[i],vzArray[i],mass_array[electron_states[i]])
                    if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 5e-24:
                        print('Uh oh')
                        quit
        scatterTheta[scatYesOrNo == 1] = np.arccos(2*np.random.rand(len(scatYesOrNo[scatYesOrNo == 1]))-1)
        scatterPhi[scatYesOrNo > 0] = np.random.rand(len(scatYesOrNo[scatYesOrNo > 0]))*2*np.pi
        vxArray,vyArray,vzArray = rotateVector(vxArray,vyArray,vzArray,scatterTheta,scatterPhi)
        randomFlightTimes[randomFlightTimes != 0] = -np.log(np.random.rand(len(randomFlightTimes[randomFlightTimes != 0])))/max_scatter_array[electron_states[randomFlightTimes != 0]]

    vxArray += e*electric_field[0]*(-timeElapsed + dt)/mass_array[electron_states]
    vyArray += e*electric_field[1]*(-timeElapsed + dt)/mass_array[electron_states]
    vzArray += e*electric_field[2]*(-timeElapsed + dt)/mass_array[electron_states]
    velocityAveragesGamma[stepNum] = np.mean(vzArray[electron_states == 0])
    velocityAveragesL_long[stepNum] = np.mean(vzArray[electron_states == 1])
    if np.isnan(velocityAveragesL_long[stepNum]) == True:
        velocityAveragesL_long[stepNum] = 0
    velocityAveragesL_tran[stepNum] = np.mean(vzArray[electron_states == 2])
    if np.isnan(velocityAveragesL_tran[stepNum]) == True:
        velocityAveragesL_tran[stepNum] = 0
    velocityAveragesX_long[stepNum] = np.mean(vzArray[electron_states == 3])
    if np.isnan(velocityAveragesX_long[stepNum]) == True:
        velocityAveragesX_long[stepNum] = 0
    velocityAveragesX_tran[stepNum] = np.mean(vzArray[electron_states == 4])
    if np.isnan(velocityAveragesX_tran[stepNum]) == True:
        velocityAveragesX_tran[stepNum] = 0
    velocityAverages[stepNum] = np.mean(vzArray)
    energyAveragesGamma[stepNum] = np.mean(energizer(vxArray[electron_states == 0],vyArray[electron_states == 0],vzArray[electron_states == 0],mass_array[electron_states[electron_states == 0]]))
    energyAveragesL_long[stepNum] = np.mean(energizer(vxArray[electron_states == 1],vyArray[electron_states == 1],vzArray[electron_states == 1],mass_array[electron_states[electron_states == 1]]))
    if np.isnan(energyAveragesL_long[stepNum]) == True:
        energyAveragesL_long[stepNum] = 0
    energyAveragesL_tran[stepNum] = np.mean(energizer(vxArray[electron_states == 2],vyArray[electron_states == 2],vzArray[electron_states == 2],mass_array[electron_states[electron_states == 2]]))
    if np.isnan(energyAveragesL_tran[stepNum]) == True:
        energyAveragesL_tran[stepNum] = 0
    energyAveragesX_long[stepNum] = np.mean(energizer(vxArray[electron_states == 3],vyArray[electron_states == 3],vzArray[electron_states == 3],mass_array[electron_states[electron_states == 3]]))
    if np.isnan(energyAveragesX_long[stepNum]) == True:
        energyAveragesX_long[stepNum] = 0
    energyAveragesX_tran[stepNum] = np.mean(energizer(vxArray[electron_states == 4],vyArray[electron_states == 4],vzArray[electron_states == 4],mass_array[electron_states[electron_states == 4]]))
    if np.isnan(energyAveragesX_tran[stepNum]) == True:
        energyAveragesX_tran[stepNum] = 0
    energyAverages[stepNum] = np.mean(energizer(vxArray,vyArray,vzArray,mass_array[electron_states]))
    print(velocityAveragesGamma[stepNum])
    print(velocityAveragesL_long[stepNum])
    print(velocityAveragesL_tran[stepNum])
    print(velocityAveragesX_long[stepNum])
    print(velocityAveragesX_tran[stepNum])

plt.figure(1)
plt.plot(times*1e12,velocityAveragesGamma/1e5,'o',label='Gamma')
plt.plot(times*1e12,velocityAveragesL_long/1e5,'o',label='L long')
plt.plot(times*1e12,velocityAveragesL_tran/1e5,'o',label='L tran')
plt.plot(times*1e12,velocityAveragesX_long/1e5,'o',label='X long')
plt.plot(times*1e12,velocityAveragesX_tran/1e5,'o',label='X tran')
plt.plot(times*1e12,velocityAverages/1e5,'o',label='Overall')
plt.legend()
plt.figure(2)
plt.plot(times*1e12,energyAveragesGamma/1.602e-19,'o',label='Gamma')
plt.plot(times*1e12,energyAveragesL_long/1.602e-19,'o',label='L long')
plt.plot(times*1e12,energyAveragesL_tran/1.602e-19,'o',label='L tran')
plt.plot(times*1e12,energyAveragesX_long/1.602e-19,'o',label='X long')
plt.plot(times*1e12,energyAveragesX_tran/1.602e-19,'o',label='X tran')
plt.plot(times*1e12,energyAverages/1.602e-19,'o',label='Overall')
plt.plot(times*1e12,0.5*mass*velocityAverages**2/1.602e-19,'o',label='Calculated from drift velocity only')
plt.legend()
plt.show()

quit
















e
