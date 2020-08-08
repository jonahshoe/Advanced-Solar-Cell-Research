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
V = 1e-6 #volume? volume of what? a unit cell? Make it one cubic centimeter? (in cubic meters)
hbar_ev = 6.58212e-16 #reduced planck constant in eV*s
hbar_j = 1.0546e-34
Temp = 300 #in kelvin
k_B = 1.38e-23
m_e = 9.109384e-31

energy_min = 1e-3 #in ev
energy_max = 0.6 # in eV
numberOfEnergies = 10000
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

def MaxwellBoltzmann(energy):
    return np.exp(-energy/(k_B*Temp))

n_i = 1e17 #intrinsic carrier concentration in GaAs in cm^-3 --  THIS IS 1E14 IN GOODNICK'S TEXTBOOK
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
acoustic_deformation_potential_gamma = 7.01*e
acoustic_deformation_potential_L = 9.2*e
acoustic_deformation_potential_X = 9.0*e
optical_deformation_potential = 3e8*e*1e2 #optical deformation potential for L valley in joules per meter
optical_energy = 0.03536*e
optical_long_freq = optical_energy/hbar_j
optical_energy_index = np.argmin(np.abs(energies-optical_energy))
mass = 0.063*m_e #from Lundstrom, definitely worth fact checking
mass_gamma = 0.063*m_e
mass_L_longitudinal = 1.9*m_e
mass_L_transverse = 0.37*m_e
# mass_L_effective = (mass_L_longitudinal*mass_L_transverse**2)**(1/3)
mass_L_effective = 0.22*m_e
mass_X_longitudinal = 1.98*m_e
mass_X_transverse = 0.27*m_e
# mass_X_effective = (mass_X_longitudinal*mass_X_transverse**2)**(1/3)
mass_X_effective = 0.58*m_e
# intervalley_deformation_potential_gammaL = 58e9*e #in J per meter
# intervalley_deformation_potential_gammaX = 80e9*e #in J per meter
# intervalley_deformation_potential_LX = 72e9*e
intervalley_deformation_potential_gammaL = 100e9*e #in J per meter
intervalley_deformation_potential_gammaX = 100e9*e #in J per meter
intervalley_deformation_potential_LX = 50e9*e
intervalley_deformation_potential_LL = 70e9*e #in J per meter
intervalley_deformation_potential_XX = 70e9*e
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

def OpticalDeformationAbsorptionScatteringRate(energy,mass):
    # return np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*phononOccupancy(optical_long_freq*hbar_j)*DensityOfStates(energy+optical_long_freq*hbar_j,mass)
    return mass**(1.5)*optical_deformation_potential**2/(np.sqrt(2)*np.pi*density*hbar_j**2*optical_energy)*phononOccupancy(optical_energy)*np.sqrt(energy+optical_energy)


def OpticalDeformationEmissionScatteringRate(energy,mass):
    output = np.zeros_like(energy)
    # output[energy > optical_energy] = np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*(phononOccupancy(optical_long_freq*hbar_j)+1)*DensityOfStates(energy[energy > optical_energy]-optical_energy,mass)
    output[energy > optical_energy] = mass**(1.5)*optical_deformation_potential**2/(np.sqrt(2)*np.pi*density*hbar_j**2*optical_energy)*(phononOccupancy(optical_energy)+1)*np.sqrt(energy[energy > optical_energy]-optical_energy)
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

def PolarOpticalAbsorptionScatteringRate(energy,mass):
    A = np.zeros_like(energy)
    A[energy > optical_energy] = e**2*optical_long_freq*(eps_low/eps_high-1)/(2*np.pi*eps_low*epsilon_0*hbar_j*np.sqrt(2*energy[energy>optical_energy]/mass))
    # A = e**2*np.sqrt(mass)*optical_long_freq/(2*np.sqrt(2)*np.pi*hbar_j**2)*(1/eps_high - 1/eps_low)*energy**(-0.5)
    N_0 = phononOccupancy(optical_energy)
    B = N_0*np.arcsinh(np.sqrt(energy/optical_energy))
    return A*B

def PolarOpticalEmissionScatteringRate(energy,mass):
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
    ionized_rate_L = IonizedImpurityScatteringRate(energies,alpha_L,mass_L_effective)
    ionized_rate_X = IonizedImpurityScatteringRate(energies,alpha_X,mass_X_effective)
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
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, L Valley'
    scatTypeEnergyShifts[scatIndex] = 0
    ii2 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, X Valley'
    scatTypeEnergyShifts[scatIndex] = 0
    ii3 = scatIndex - 1
    scatIndex += 1

if acoustic_deformation_potential_scattering == 1:
    acoustic_rates_gamma = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_gamma,mass_gamma) #double it to take absorption + emission into account
    acoustic_rates_L = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_L,mass_L_effective)
    acoustic_rates_X = AcousticDeformationElasticScatteringRate(energies,acoustic_deformation_potential_X,mass_X_effective)
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
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, L Valley'
    scatTypeEnergyShifts[scatIndex] = 0
    adp2 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, X Valley'
    scatTypeEnergyShifts[scatIndex] = 0
    adp3 = scatIndex - 1
    scatIndex += 1

if optical_deformation_potential_scattering == 1:
    optical_absorption_rates_gamma = OpticalDeformationAbsorptionScatteringRate(energies,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    odp1 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_gamma = OpticalDeformationEmissionScatteringRate(energies,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    odp2 = scatIndex - 1
    scatIndex += 1

    optical_absorption_rates_L = OpticalDeformationAbsorptionScatteringRate(energies,mass_L_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, L Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    odp3 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_L = OpticalDeformationEmissionScatteringRate(energies,mass_L_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, L Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    odp4 = scatIndex - 1
    scatIndex += 1

    optical_absorption_rates_X = OpticalDeformationAbsorptionScatteringRate(energies,mass_X_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption, X Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    odp5 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates_X = OpticalDeformationEmissionScatteringRate(energies,mass_X_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission, X Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    odp6 = scatIndex - 1
    scatIndex += 1

if intervalley_deformation_potential_scattering == 1:

    rate_gammaL_abs = IntervalleyAbsorptionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to L'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL-energy_separation_gammaL
    ivgL1 = scatIndex - 1
    scatIndex += 1
    rate_gammaL_emi = IntervalleyEmissionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaL_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to L'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL-energy_separation_gammaL
    ivgL2 = scatIndex - 1
    scatIndex += 1

    rate_gammaX_abs = IntervalleyAbsorptionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, Gamma to X'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX-energy_separation_gammaX
    ivgX1 = scatIndex - 1
    scatIndex += 1
    rate_gammaX_emi = IntervalleyEmissionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gammaX_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, Gamma to X'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX-energy_separation_gammaX
    ivgX2 = scatIndex - 1
    scatIndex += 1

    rate_Lgamma_abs = IntervalleyAbsorptionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=-energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Lgamma_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to Gamma'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaL+energy_separation_gammaL
    ivLg1 = scatIndex - 1
    scatIndex += 1
    rate_Lgamma_emi = IntervalleyEmissionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaL,intervalley_phonon_energy_gammaL,energy_separation=-energy_separation_gammaL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Lgamma_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to Gamma'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaL+energy_separation_gammaL
    ivLg2 = scatIndex - 1
    scatIndex += 1

    rate_LX_abs = IntervalleyAbsorptionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to X'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX
    ivLX1 = scatIndex - 1
    scatIndex += 1
    rate_LX_emi = IntervalleyEmissionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LX_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to X'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX-energy_separation_LX
    # scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX
    ivLX2 = scatIndex - 1
    scatIndex += 1

    rate_LL_abs = IntervalleyAbsorptionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, L to L'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LL
    ivLL1 = scatIndex - 1
    scatIndex += 1
    rate_LL_emi = IntervalleyEmissionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_LL,intervalley_phonon_energy_LL)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_LL_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, L to L'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LL
    ivLL2 = scatIndex - 1
    scatIndex += 1

    rate_Xgamma_abs = IntervalleyAbsorptionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=-energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Xgamma_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to Gamma'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_gammaX+energy_separation_gammaX
    ivXg1 = scatIndex - 1
    scatIndex += 1
    rate_Xgamma_emi = IntervalleyEmissionScatteringRate(energies,mass_gamma,intervalley_deformation_potential_gammaX,intervalley_phonon_energy_gammaX,energy_separation=-energy_separation_gammaX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_Xgamma_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to Gamma'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_gammaX+energy_separation_gammaX
    ivXg2 = scatIndex - 1
    scatIndex += 1

    rate_XL_abs = IntervalleyAbsorptionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to L'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_LX+energy_separation_LX
    ivXL1 = scatIndex - 1
    scatIndex += 1
    rate_XL_emi = IntervalleyAbsorptionScatteringRate(energies,mass_L_effective,intervalley_deformation_potential_LX,intervalley_phonon_energy_LX,energy_separation=-energy_separation_LX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XL_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to L'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_LX+energy_separation_LX
    ivXL2 = scatIndex - 1
    scatIndex += 1

    rate_XX_abs = IntervalleyAbsorptionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_abs]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, X to X'
    scatTypeEnergyShifts[scatIndex] = intervalley_phonon_energy_XX
    ivXX1 = scatIndex - 1
    scatIndex += 1
    rate_XX_emi = IntervalleyEmissionScatteringRate(energies,mass_X_effective,intervalley_deformation_potential_XX,intervalley_phonon_energy_XX)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_XX_emi]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, X to X'
    scatTypeEnergyShifts[scatIndex] = -intervalley_phonon_energy_XX
    ivXX2 = scatIndex - 1
    scatIndex += 1

if polar_optical_scattering == 1:

    polar_optical_absorption_rates_gamma = PolarOpticalAbsorptionScatteringRate(energies,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    popG1 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_gamma = PolarOpticalEmissionScatteringRate(energies,mass_gamma)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_gamma]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, Gamma Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    popG2 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_L = PolarOpticalAbsorptionScatteringRate(energies,mass_L_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, L Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    popL1 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_L = PolarOpticalEmissionScatteringRate(energies,mass_L_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_L]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, L Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    popL2 = scatIndex - 1
    scatIndex += 1

    polar_optical_absorption_rates_X = PolarOpticalAbsorptionScatteringRate(energies,mass_X_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_absorption_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Absorption, X Valley'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    popX1 = scatIndex - 1
    scatIndex += 1

    polar_optical_emission_rates_X = PolarOpticalEmissionScatteringRate(energies,mass_X_effective)
    scatteringRateTable = np.concatenate((scatteringRateTable,[polar_optical_emission_rates_X]),axis=0)
    scatTypeLabels[scatIndex] = 'Polar Optical Scattering, Emission, X Valley'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    popX2 = scatIndex - 1
    scatIndex += 1

if plot_stuff == 1:
    plt.figure(0)
    plt.title('Ionized Impurity')
    plt.plot(energies/e,scatteringRateTable[ii1],label='gamma')
    plt.plot(energies/e,scatteringRateTable[ii2],label='L')
    plt.plot(energies/e,scatteringRateTable[ii3],label="X")
    plt.legend()
    plt.figure(1)
    plt.title('Acoustic Deformation Potential')
    plt.plot(energies/e,scatteringRateTable[adp1],label="gamma")
    plt.plot(energies/e,scatteringRateTable[adp2],label='L')
    plt.plot(energies/e,scatteringRateTable[adp3],label="L")
    plt.legend()
    plt.figure(2)
    plt.title('Optical Deformation Potential')
    plt.plot(energies/e,scatteringRateTable[odp1],label='gamma, absorption')
    plt.plot(energies/e,scatteringRateTable[odp2],label='gamma, emission')
    plt.plot(energies/e,scatteringRateTable[odp3],label='L, absorption')
    plt.plot(energies/e,scatteringRateTable[odp4],label='L, emission')
    plt.plot(energies/e,scatteringRateTable[odp5],label='X, absorption')
    plt.plot(energies/e,scatteringRateTable[odp6],label='X, emission')
    plt.legend()
    plt.figure(3)
    plt.title('Intervalley Scattering From Gamma Valley')
    plt.plot(energies/e,scatteringRateTable[ivgL1],label='absorption, gamma to L')
    plt.plot(energies/e,scatteringRateTable[ivgL2],label='absorption, gamma to L')
    plt.plot(energies/e,scatteringRateTable[ivgX1],label='emission, gamma to X')
    plt.plot(energies/e,scatteringRateTable[ivgX2],label='emission, gamma to X')
    plt.legend()
    plt.figure(4)
    plt.title('intervalley Scattering From L Valley')
    plt.plot(energies/e,scatteringRateTable[ivLg1],label='absorption, L to gamma')
    plt.plot(energies/e,scatteringRateTable[ivLg2],label='emission, L to gamm')
    plt.plot(energies/e,scatteringRateTable[ivLX1],label='absorption, L to X')
    plt.plot(energies/e,scatteringRateTable[ivLX2],label='emission, L to X')
    plt.plot(energies/e,scatteringRateTable[ivLL1],label='absorption, L to L')
    plt.plot(energies/e,scatteringRateTable[ivLL2],label='emission, L to L')
    plt.legend()
    plt.figure(5)
    plt.title('intervalley Scattering From X Valley')
    plt.plot(energies/e,scatteringRateTable[ivXg1],label='absorption, X to gamma')
    plt.plot(energies/e,scatteringRateTable[ivXg2],label='emission, X to gamma')
    plt.plot(energies/e,scatteringRateTable[ivXL1],label='absorption, X to L')
    plt.plot(energies/e,scatteringRateTable[ivXL2],label='emission, X to L')
    plt.plot(energies/e,scatteringRateTable[ivXX1],label='absorption, X to X')
    plt.plot(energies/e,scatteringRateTable[ivXX2],label='emission, X to X')
    plt.legend()
    plt.figure(6)
    plt.title('Polar Optical Scattering')
    plt.plot(energies/e,scatteringRateTable[popG1],label='absorption, gamma')
    plt.plot(energies/e,scatteringRateTable[popG2],label='emission, gamma')
    plt.plot(energies/e,scatteringRateTable[popL1],label='absorption, L')
    plt.plot(energies/e,scatteringRateTable[popL2],label='emission, L')
    plt.plot(energies/e,scatteringRateTable[popX1],label='absorption, X')
    plt.plot(energies/e,scatteringRateTable[popX2],label='emission, X')
    plt.yscale('log')
    plt.legend()
    plt.show()
    quit

scatteringSums = np.zeros_like(scatteringRateTable)
for i in range(len(scatteringRateTable)):
    if i == 0:
        scatteringSums[i] = scatteringRateTable[i]
    if i > 0:
        scatteringSums[i] = scatteringRateTable[i]+scatteringSums[i-1]
max_scatter = np.max(scatteringSums[-1])

scatTypeConverterGamma = np.array([ii1,adp1,odp1,odp2,ivgL1,ivgL2,ivgX1,ivgX2,popG1,popG2])
scatteringRateTableGamma = scatteringRateTable[scatTypeConverterGamma]
scatteringRateTableGamma[scatTypeConverterGamma == ivgL1] *= 4
scatteringRateTableGamma[scatTypeConverterGamma == ivgL2] *= 4
scatteringRateTableGamma[scatTypeConverterGamma == ivgX1] *= 3
scatteringRateTableGamma[scatTypeConverterGamma == ivgX2] *= 3
scatteringSumsGamma = np.zeros_like(scatteringRateTableGamma)
for i in range(len(scatteringRateTableGamma)):
    if i == 0:
        scatteringSumsGamma[i] = scatteringRateTableGamma[i]
    if i > 0:
        scatteringSumsGamma[i] = scatteringRateTableGamma[i]+scatteringSumsGamma[i-1]
max_scatter_gamma = np.max(scatteringSumsGamma[-1])

scatTypeConverterL = np.array([ii2,adp2,odp3,odp4,ivLg1,ivLg2,ivLL1,ivLL2,ivLX1,ivLX2,popL1,popL2])
scatteringRateTableL = scatteringRateTable[scatTypeConverterL]
scatteringRateTableL[scatTypeConverterL == ivLL1] *= 3
scatteringRateTableL[scatTypeConverterL == ivLL2] *= 3
scatteringRateTableL[scatTypeConverterL == ivLX1] *= 3
scatteringRateTableL[scatTypeConverterL == ivLX2] *= 3
scatteringSumsL = np.zeros_like(scatteringRateTableL)
for i in range(len(scatteringRateTableL)):
    if i == 0:
        scatteringSumsL[i] = scatteringRateTableL[i]
    if i > 0:
        scatteringSumsL[i] = scatteringRateTableL[i]+scatteringSumsL[i-1]
max_scatter_L = np.max(scatteringSumsL[-1])

scatTypeConverterX = np.array([ii3,adp3,odp5,odp6,ivXg1,ivXg2,ivXL1,ivXL2,ivXX1,ivXX2,popX1,popX2])
scatteringRateTableX = scatteringRateTable[scatTypeConverterX]
scatteringRateTableX[scatTypeConverterX == ivXX1] *= 2
scatteringRateTableX[scatTypeConverterX == ivXX2] *= 2
scatteringRateTableX[scatTypeConverterX == ivXL1] *= 4
scatteringRateTableX[scatTypeConverterX == ivXL2] *= 4
scatteringSumsX = np.zeros_like(scatteringRateTableX)
for i in range(len(scatteringRateTableX)):
    if i == 0:
        scatteringSumsX[i] = scatteringRateTableX[i]
    if i > 0:
        scatteringSumsX[i] = scatteringRateTableX[i]+scatteringSumsX[i-1]
max_scatter_X = np.max(scatteringSumsX[-1])


def energizer(kx,ky,kz,alpha,mass):
    """Returns energy (in Joules) of a particle with velocity components vx,vy,vz."""
    return (np.sqrt(1 + 4*alpha/e*hbar_j**2*(kx**2+ky**2+kz**2)/(2*mass) ) - 1)/(2*alpha/e)

def rotateVector(vx,vy,vz,theta,phi):
    if type(theta) == np.ndarray:
        vx_rotated = np.zeros_like(vx)
        vy_rotated = np.zeros_like(vy)
        vz_rotated = np.zeros_like(vz)
        for i in range(len(theta)):
            vMagnitude = np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2)
            if vx[i] != 0:
                framePhi = np.arctan2(vy[i],vx[i])
            else:
                framePhi = 0
            if vz[i] != 0:
                frameTheta = np.arctan2(np.sqrt(vx[i]**2+vy[i]**2),vz[i])
            else:
                frameTheta = 0
            vx_temp = vMagnitude*np.sin(-theta[i])*np.cos(phi[i])
            vy_temp = vMagnitude*np.sin(-theta[i])*np.sin(phi[i])
            vz_temp = vMagnitude*np.cos(-theta[i])
            vx_rotated[i] = np.cos(framePhi)*np.cos(frameTheta)*vx_temp - np.sin(framePhi)*vy_temp + np.cos(framePhi)*np.sin(frameTheta)*vz_temp
            vy_rotated[i] = np.sin(framePhi)*np.cos(frameTheta)*vx_temp + np.cos(framePhi)*vy_temp + np.sin(framePhi)*np.sin(frameTheta)*vz_temp
            vz_rotated[i] = -np.sin(frameTheta)*vx_temp + np.cos(frameTheta)*vz_temp
    else:
        vMagnitude = np.sqrt(vx**2+vy**2+vz**2)
        framePhi = np.arctan2(vy,vx)
        frameTheta = np.arctan2(np.sqrt(vx**2+vy**2),vz)
        vx_temp = vMagnitude*np.sin(theta)*np.cos(phi)
        vy_temp = vMagnitude*np.sin(theta)*np.sin(phi)
        vz_temp = vMagnitude*np.cos(theta)
        vx_rotated = np.cos(framePhi)*np.cos(frameTheta)*vx_temp - np.sin(framePhi)*vy_temp + np.cos(framePhi)*np.sin(frameTheta)*vz_temp
        vy_rotated = np.sin(framePhi)*np.cos(frameTheta)*vx_temp + np.cos(framePhi)*vy_temp + np.sin(framePhi)*np.sin(frameTheta)*vz_temp
        vz_rotated = -np.sin(frameTheta)*vx_temp + np.cos(frameTheta)*vz_temp

    return vx_rotated,vy_rotated,vz_rotated

def RandomThetaSelectorIonizedImpurityScattering(energy,mass):
    tryAgain = 1
    ksquared = 2*mass*energy/hbar_j**2
    max = (2*ksquared*screeninglength_meters**2)**2
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < (1-np.cos(randomTheta)+(2*ksquared*screeninglength_meters**2)**(-1))**(-2):
            tryAgain = 0
            return randomTheta

def polarOpticalAbsorptionRandomTheta():
    tryAgain = 1
    max = 1e4
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < np.sin(randomTheta)/(1-np.cos(randomTheta)):
            tryAgain = 0
            return randomTheta

def polarOpticalEmissionRandomTheta():
    tryAgain = 1
    max = 1e4
    tryAgain = 1
    while tryAgain == 1:
        r1 = np.random.rand()
        r2 = np.random.rand()
        randomTheta = r1*np.pi
        if r2*max < np.sin(randomTheta)/(1 - np.cos(randomTheta)):
            tryAgain = 0
            return randomTheta


def MaxwellBoltzmannEnergyDistribution(numberOfNeededEnergies):
    outputList = []
    tryAgain = 1
    while len(outputList) < numberOfNeededEnergies:
        x = np.random.rand()*1e21
        energyOutput = np.random.rand()*0.2*e
        y = np.exp(-energyOutput/(k_B*Temp))/(k_B*Temp)
        if x < y:
            outputList.append(energyOutput)
    output = np.asarray(outputList)
    return output

# energy_mesh,theta_mesh = np.meshgrid(np.linspace(1e-4,1,1000)*e,np.linspace(0,np.pi,1000))
#
# def CarrierCarrier(energy,mass):
#     kminusk0 = 2*mass/hbar_j**2*(energy_mesh + energy - 2*np.sqrt(energy_mesh*energy)*np.cos(theta_mesh))
#     return np.max(kminusk0/(kminusk0**2+1/screeninglength_meters**2))

##### Generate random flight time before collision #####

time_step_count = 120
particle_count = 500
start_time = 0
max_time = 6e-12
dt = (max_time-start_time)/time_step_count
times = np.linspace(start_time,max_time,time_step_count)

efield_min = 0
efield_max = 50e5
efield_count = 100
efield_array = np.linspace(efield_min,efield_max,efield_count)
electric_field = np.array([0,0,0])
mass_array = np.array([mass_gamma,mass_L_effective,mass_X_effective])
alpha_array = np.array([alpha_gamma,alpha_L,alpha_X])
max_scatter_array = np.array([max_scatter_gamma,max_scatter_L,max_scatter_X])

timeAveragedVelocities = np.zeros([efield_count])
timeAveragedEnergies = np.zeros([efield_count])
megaVelocityAverages = np.zeros([efield_count,time_step_count])
megaEnergyAverages = np.zeros([efield_count,time_step_count])

for eFieldNum in range(efield_count):
    startingEnergies = MaxwellBoltzmannEnergyDistribution(particle_count)
    startingThetas = np.random.rand(particle_count)*np.pi
    startingPhis = np.random.rand(particle_count)*2*np.pi
    startingKMagnitudes = np.sqrt(2*mass*startingEnergies*(1+alpha_gamma*startingEnergies/e))/hbar_j
    kxArray = startingKMagnitudes*np.cos(startingPhis)*np.sin(startingThetas)
    kyArray = startingKMagnitudes*np.sin(startingPhis)*np.sin(startingThetas)
    kzArray = startingKMagnitudes*np.cos(startingThetas)
    vxArray = hbar_j*kxArray/(mass*(1+2*alpha_gamma*startingEnergies))
    vyArray = hbar_j*kyArray/(mass*(1+2*alpha_gamma*startingEnergies))
    vzArray = hbar_j*kzArray/(mass*(1+2*alpha_gamma*startingEnergies))
    electric_field[2] = efield_array[eFieldNum]
    print('Starting on E-field Strength: ', electric_field[2]/1e5, ' kV/cm')
    electron_states = np.zeros(particle_count).astype(int)
    r = np.random.rand(particle_count)
    electron_states[r < MaxwellBoltzmann(0.29*e-startingEnergies)] = 1
    electron_states[r < MaxwellBoltzmann(0.48*e-startingEnergies)] = 2

    energyAveragesGamma = np.zeros([time_step_count])
    energyAveragesL= np.zeros([time_step_count])
    energyAveragesX = np.zeros([time_step_count])
    energyAverages = np.zeros([time_step_count])
    velocityAveragesGamma = np.zeros([time_step_count])
    velocityAveragesL = np.zeros([time_step_count])
    velocityAveragesX = np.zeros([time_step_count])
    velocityAverages = np.zeros([time_step_count])
    totalEnergyShift = np.zeros([particle_count])
    totalEnergyGain = np.zeros([particle_count])

    scatCount = np.zeros(len(scatteringRateTable)).astype(int)
    scatCount2 = scatCount.copy()
    velocityAverages[0] = np.mean(vzArray)
    velocityAveragesGamma[0] = np.mean(vzArray[electron_states == 0])
    velocityAveragesL[0] = 0
    velocityAveragesX[0] = 0
    energyAverages[0] = np.mean(energizer(kxArray,kyArray,kzArray,alpha_array[electron_states],mass_array[electron_states]))
    energyAveragesGamma[0] = energyAverages[0]
    energyAveragesL[0] = 0
    energyAveragesX[0] = 0
    print(np.mean(vzArray))
    for stepNum in range(time_step_count-1):
        # print(scatCount-scatCount2)
        scatCount2 = scatCount.copy()
        print('Starting on time interval ', stepNum, ' out of ', time_step_count, '.')
        timeElapsed = np.zeros(particle_count)
        randomFlightTimes = -np.log(np.random.rand(particle_count))/max_scatter_array[electron_states]
        while np.max(randomFlightTimes) > 0:
            randomFlightTimes[randomFlightTimes+timeElapsed > dt] = 0
            timeElapsed += randomFlightTimes

            kxArray = kxArray + e*electric_field[0]*randomFlightTimes/hbar_j
            kyArray = kyArray + e*electric_field[1]*randomFlightTimes/hbar_j
            kzArray = kzArray + e*electric_field[2]*randomFlightTimes/hbar_j
            currentEnergies = energizer(kxArray,kyArray,kzArray,alpha_array[electron_states],mass_array[electron_states])
            currentEnergyIndices = np.zeros([particle_count]).astype(int)
            randomScats = np.zeros(particle_count)
            randomScats[randomFlightTimes != 0] = np.random.rand(len(randomFlightTimes[randomFlightTimes != 0]))*max_scatter
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
                        old_alpha = alpha_array[electron_states[i]]
                        if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, Gamma Valley':
                            scatYesOrNo[i] = 2
                            scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, Gamma Valley':
                            scatterTheta[i] = polarOpticalAbsorptionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, Gamma Valley':
                            scatterTheta[i] = polarOpticalEmissionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to L' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to L':
                            electron_states[i] = 1
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, Gamma to X' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, Gamma to X':
                            electron_states[i] = 2
                        energyShift = scatTypeEnergyShifts[scatteringType[i]]
                        totalEnergyShift[i] += energyShift
                        if currentEnergies[i]+energyShift < 0:
                            if currentEnergies[i]+energyShift > -1e-23:
                                kxArray[i] = 0
                                kyArray[i] = 0
                                kzArray[i] = 0
                            else:
                                print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                                quit
                        else:
                            kxArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kyArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kzArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                        newEnergy = energizer(kxArray[i],kyArray[i],kzArray[i],alpha_array[electron_states[i]],mass_array[electron_states[i]])
                        if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 1e-23:
                            print('Uh oh')
                            quit
                elif electron_states[i] == 1:
                    for scatNum in range(len(scatteringSumsL)):
                        if randomScats[i] < scatteringSumsL[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                            scatteringType[i] = scatTypeConverterL[scatNum]+1
                            scatYesOrNo[i] = 1
                            scatCount[scatTypeConverterL[scatNum]] += 1
                            break
                        elif scatNum == len(scatteringSumsL)-1:
                            scatYesOrNo[i] = 0
                            scatteringType[i] = 0
                    if scatteringType[i] != 0:
                        old_mass = mass_array[electron_states[i]]
                        old_alpha = alpha_array[electron_states[i]]
                        if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, L Valley':
                            scatYesOrNo[i] = 2
                            scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, L Valley':
                            scatterTheta[i] = polarOpticalAbsorptionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, L Valley':
                            scatterTheta[i] = polarOpticalEmissionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to Gamma':
                            electron_states[i] = 0
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, L to X' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, L to X':
                            electron_states[i] = 2
                        energyShift = scatTypeEnergyShifts[scatteringType[i]]
                        totalEnergyShift[i] += energyShift
                        if currentEnergies[i]+energyShift < 0:
                            if currentEnergies[i]+energyShift > -1e-23:
                                kxArray[i] = 0
                                kyArray[i] = 0
                                kzArray[i] = 0
                            else:
                                print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                                quit
                        else:
                            kxArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kyArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kzArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                        newEnergy = energizer(kxArray[i],kyArray[i],kzArray[i],alpha_array[electron_states[i]],mass_array[electron_states[i]])
                        if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 1e-23:
                            print('Uh oh')
                            quit
                elif electron_states[i] == 2:
                    for scatNum in range(len(scatteringSumsX)):
                        if randomScats[i] < scatteringSumsX[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                            scatteringType[i] = scatTypeConverterX[scatNum]+1
                            scatYesOrNo[i] = 1
                            scatCount[scatTypeConverterX[scatNum]] += 1
                            break
                        elif scatNum == len(scatteringSumsX)-1:
                            scatYesOrNo[i] = 0
                            scatteringType[i] = 0
                    if scatteringType[i] != 0:
                        old_mass = mass_array[electron_states[i]]
                        old_alpha = alpha_array[electron_states[i]]
                        if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, X Valley':
                            scatYesOrNo[i] = 2
                            scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Absorption, X Valley':
                            scatterTheta[i] = polarOpticalAbsorptionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Polar Optical Scattering, Emission, X Valley':
                            scatterTheta[i] = polarOpticalEmissionRandomTheta()
                            scatYesOrNo[i] = 2
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to Gamma' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to Gamma':
                            electron_states[i] = 0
                        if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, X to L' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, X to L':
                            electron_states[i] = 1
                        energyShift = scatTypeEnergyShifts[scatteringType[i]]
                        totalEnergyShift[i] += energyShift
                        if currentEnergies[i]+energyShift < 0:
                            if currentEnergies[i]+energyShift > -1e-23:
                                kxArray[i] = 0
                                kyArray[i] = 0
                                kzArray[i] = 0
                            else:
                                print('Error: check on scattering type ', scatTypeLabels[scatteringType[i]])
                                quit
                        else:
                            kxArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kyArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                            kzArray[i] *= np.sqrt( (mass_array[electron_states[i]]/old_mass) * ((currentEnergies[i]+energyShift)/currentEnergies[i]) * ((1 + alpha_array[electron_states[i]]*(currentEnergies[i] + energyShift)/e) / (1 + old_alpha*currentEnergies[i]/e)) )
                        newEnergy = energizer(kxArray[i],kyArray[i],kzArray[i],alpha_array[electron_states[i]],mass_array[electron_states[i]])
                        if np.abs(newEnergy - (currentEnergies[i]+energyShift)) > 1e-23:
                            print('Uh oh')
                            quit
            scatterTheta[scatYesOrNo == 1] = np.arccos(2*np.random.rand(len(scatYesOrNo[scatYesOrNo == 1]))-1)
            scatterPhi[scatYesOrNo > 0] = np.random.rand(len(scatYesOrNo[scatYesOrNo > 0]))*2*np.pi
            kxArray,kyArray,kzArray = rotateVector(kxArray,kyArray,kzArray,scatterTheta,scatterPhi)
            randomFlightTimes[randomFlightTimes != 0] = -np.log(np.random.rand(len(randomFlightTimes[randomFlightTimes != 0])))/max_scatter

        kxArray = kxArray + e*electric_field[0]*(-timeElapsed+dt)/hbar_j
        kyArray = kyArray + e*electric_field[1]*(-timeElapsed+dt)/hbar_j
        kzArray = kzArray + e*electric_field[2]*(-timeElapsed+dt)/hbar_j
        currentEnergies = energizer(kxArray,kyArray,kzArray,alpha_array[electron_states],mass_array[electron_states])
        vxArray = hbar_j*kxArray/(mass_array[electron_states]*(1 + 2*alpha_array[electron_states]*currentEnergies/e))
        vyArray = hbar_j*kyArray/(mass_array[electron_states]*(1 + 2*alpha_array[electron_states]*currentEnergies/e))
        vzArray = hbar_j*kzArray/(mass_array[electron_states]*(1 + 2*alpha_array[electron_states]*currentEnergies/e))

        velocityAveragesGamma[stepNum+1] = np.mean(vzArray[electron_states == 0])
        if np.isnan(np.mean(vzArray[electron_states == 1])) == False:
            velocityAveragesL[stepNum+1] = np.mean(vzArray[electron_states == 1])
        if np.isnan(np.mean(vzArray[electron_states == 2])) == False:
            velocityAveragesX[stepNum+1] = np.mean(vzArray[electron_states == 2])

        energyAveragesGamma[stepNum+1] = np.mean(energizer(kxArray[electron_states == 0],kyArray[electron_states == 0],kzArray[electron_states == 0],alpha_array[electron_states[electron_states == 0]],mass_array[electron_states[electron_states == 0]]))

        if np.isnan(np.mean(energizer(kxArray[electron_states == 1],kyArray[electron_states == 1],kzArray[electron_states == 1],alpha_array[electron_states[electron_states == 1]],mass_array[electron_states[electron_states == 1]]))) == False:
            energyAveragesL[stepNum+1] = np.mean(energizer(kxArray[electron_states == 1],kyArray[electron_states == 1],kzArray[electron_states == 1],alpha_array[electron_states[electron_states == 1]],mass_array[electron_states[electron_states == 1]]))

        if np.isnan(np.mean(energizer(kxArray[electron_states == 2],kyArray[electron_states == 2],kzArray[electron_states == 2],alpha_array[electron_states[electron_states == 2]],mass_array[electron_states[electron_states == 2]]))) == False:
            energyAveragesX[stepNum+1] = np.mean(energizer(kxArray[electron_states == 2],kyArray[electron_states == 2],kzArray[electron_states == 2],alpha_array[electron_states[electron_states == 2]],mass_array[electron_states[electron_states == 2]]))

        velocityAverages[stepNum+1] = np.mean(vzArray)
        energyAverages[stepNum+1] = np.mean(energizer(kxArray,kyArray,kzArray,alpha_array[electron_states],mass_array[electron_states]))
        # if stepNum+1 == time_step_count:
        print(' ')
        print(' ')
        print(velocityAverages[stepNum+1])
        print(' ')
        print(' ')
        print('NUMBER OF GAMMA ELECTRONS: ', len(electron_states[electron_states == 0]))
        print('NUMBER OF L ELECTRONS: ', len(electron_states[electron_states == 1]))
        print('NUMBER OF X ELECTRONS: ', len(electron_states[electron_states == 2]))
        print(' ')
        print(scatCount)
        print(' ')
    timeAveragedVelocities[eFieldNum] = np.mean(velocityAverages[times > 0.9*times])
    timeAveragedEnergies[eFieldNum] = np.mean(energyAverages[times > 0.9*times])
    megaVelocityAverages[eFieldNum] = velocityAverages.copy()
    megaEnergyAverages[eFieldNum] = energyAverages.copy()

# velocityFileName = 'velocityAverages' + str((electric_field[2]/1e5).astype(int)) + 'kVpercmEzFieldEquilibrium'
# np.save(velocityFileName,velocityAverages,allow_pickle=True)
# energyFileName = 'energyAverages' + str((electric_field[2]/1e5).astype(int)) + 'kVpercmEzFieldEquilibrium'
# np.save(energyFileName,energyAverages,allow_pickle=True)
plt.figure(1)
# plt.plot(times*1e12,velocityAveragesGamma/1e5,'o',label='Gamma')
# plt.plot(times*1e12,velocityAveragesL/1e5,'o',label='L')
# plt.plot(times*1e12,velocityAveragesX/1e5,'o',label='X')
plt.plot(efield_array/1e5,timeAveragedVelocities/1e5,'o',label='Overall')
plt.legend()
plt.figure(2)
# plt.plot(times*1e12,energyAveragesGamma/1.602e-19,'o',label='Gamma')
# plt.plot(times*1e12,energyAveragesL/1.602e-19,'o',label='L')
# plt.plot(times*1e12,energyAveragesX/1.602e-19,'o',label='X')
plt.plot(efield_array/1e5,timeAveragedEnergies/e,'o',label='Overall')
# plt.plot(times*1e12,0.5*mass*velocityAverages**2/1.602e-19,'o',label='Calculated from drift velocity only')
plt.legend()
plt.show()

quit
















e
