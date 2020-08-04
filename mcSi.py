import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

ionized_impurity_scattering = 1
acoustic_deformation_potential_scattering = 1
optical_deformation_potential_scattering = 1
intervalley_deformation_potential_scattering = 1
silicon = 1
GaAs = 0

e = 1.602e-19 #charge of electron in eV
V = 1 #volume? volume of what? a unit cell? Make it one cubic centimeter?
hbar_ev = 6.58212e-16 #reduced planck constant in eV*s
hbar_j = 1.0546e-34
Temp = 50 #20 deg celsius in kelvin
k_B = 1.38e-23
m_e = 9.109384e-31

energy_min = 1e-6 #in ev
energy_max = 1 # in eV
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

if silicon == 1:
    # n_i = 1e16 #typical doping density per cubic centimeter in silicon semiconductors
    n_i = 1.45e15 #from https://www.keysight.com/upload/cmc_upload/All/EE_REF_PROPERTIES_Si_Ga.pdf (in cm^-3)
    n_i *= 1e6 #convert to m^-3
    Z = 1 #net charge of ionized impurity
    mass = 0.98*m_e #effective longitudinal mass of the electron in silicon in kilograms
    eps = 11.9 #relative permittivity of pure silicon
    epsilon_0 = 8.8541878128e-12 #in farads per meter
    # screeninglength = 2.5e7 #inverse screening length in silicon in cm^-1 for 1e18 cm^-3 n-doping concentration
    # screeninglength_meters = screeninglength*1e-2
    # screeninglength_meters = 24e-6 #obtained from https://www.keysight.com/upload/cmc_upload/All/EE_REF_PROPERTIES_Si_Ga.pdf
    screeninglength_meters = np.sqrt(eps*epsilon_0*k_B*Temp/e**2/n_i)
    u_l = 8433 #speed of sound in silicon in m/s at 20 degrees celsius
    density = 2.3e3 #approximate density of silicon, rounded up to account for n-doping, in kg/m^3
    c_l = density*u_l**2
    E_1 = 9*e #acoustic deformation potential for silicon in eV
    acoustic_deformation_potential = E_1
    alpha = 0.5 #nonparabolicity factor for silicon
    optical_energy = 0.063*e #from Lundstrom's textbook, page 114, in Joules
    optical_long_freq = optical_energy/hbar_j
    optical_energy_index = np.argmin(np.abs(energies-optical_energy))
    optical_deformation_potential = 2.2e10*e
    occupancy_SiOpticalLong = phononOccupancy(hbar_j*optical_long_freq)
    phonon_occupancy = occupancy_SiOpticalLong.copy()
    mass_longitudinal = 0.916*m_e
    mass_transverse = 0.19*m_e
    mass_average = (mass_transverse**2*mass_longitudinal)**(1/3)
    Dg_TA = 5e9*e
    Dg_LA = 8e9*e
    Dg_LO = 11e10*e
    Df_TA = 3e9*e
    Df_LA = 2e10*e
    Df_TO = 2e10*e
    Eg_TA = 0.012*e
    Eg_LA = 0.019*e
    Eg_LO = 0.062*e
    Ef_TA = 0.019*e
    Ef_LA = 0.047*e
    Ef_TO = 0.059*e
### some of these constants were obtained from Sze's textbook ###


##### Impact ionization rate calculation here #####

def IonizedImpurityScatteringRate(energy,mass):
    C = n_i*e**4/(32*np.sqrt(2*mass)*np.pi*(eps*epsilon_0)**2)
    gamma_squared = 8*mass*energy*screeninglength_meters**2/hbar_j**2
    return C*gamma_squared**2/(1+gamma_squared)*energy**(-1.5)

def AcousticDeformationElasticScatteringRate(energy,mass):
    # return np.sqrt(2*mass**3)*(E_1)**2*k_B*Temp/(np.pi*hbar_j**4*c_l)*np.sqrt(energy)
    return np.pi*acoustic_deformation_potential**2*k_B*Temp/(hbar_j*c_l)*DensityOfStates(energy,mass)


def OpticalDeformationAbsorptionScatteringRate(energy,mass):
    return np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*phononOccupancy(optical_long_freq*hbar_j)*DensityOfStates(energy+optical_energy,mass)


def OpticalDeformationEmissionScatteringRate(energy,mass):
    output = np.zeros_like(energy)
    output[energy > optical_energy] = np.pi*(optical_deformation_potential)**2/(2*density*optical_long_freq)*(phononOccupancy(optical_long_freq*hbar_j)+1)*DensityOfStates(energy[energy > optical_energy]-optical_energy,mass)
    return output


def EquivalentIntervalleyAbsorptionScatteringRate(energy,m_j,D_ij,E_ij):
    # C1 = m_j**1.5*(D_ij)**2/(np.sqrt(2)*np.pi*density*hbar_j**2*E_ij)
    # C2 = np.sqrt(energy+E_ij)*phononOccupancy(E_ij)
    # return C1*C2
    return np.pi*D_ij**2/(2*density*E_ij/hbar_j)*phononOccupancy(E_ij)*DensityOfStates(energy+E_ij,m_j)


def EquivalentIntervalleyEmissionScatteringRate(energy,m_j,D_ij,E_ij):
    # C1 = m_j**1.5*(D_ij)**2/(np.sqrt(2)*np.pi*density*hbar_j**2*E_ij)
    # C2 = np.zeros_like(energy)
    # C2[energy-E_ij > 0] = np.sqrt(energy[energy-E_ij > 0]-E_ij)*(phononOccupancy(E_ij)+1)
    # return C1*C2
    output = np.zeros_like(energy)
    output[energy > E_ij] = np.pi*D_ij**2/(2*density*E_ij/hbar_j)*(phononOccupancy(E_ij)+1)*DensityOfStates(energy[energy > E_ij]-E_ij,m_j)
    return output

scatIndex = 1

if ionized_impurity_scattering == 1:
    ionized_rate_long = IonizedImpurityScatteringRate(energies,mass_longitudinal)
    ionized_rate_tran = IonizedImpurityScatteringRate(energies,mass_transverse)
    if scatIndex == 1:
        scatteringRateTable = np.array([ionized_rate_long])
        scatTypeLabels = {scatIndex: 'Ionized Impurity, Longitudinal'}
        scatTypeEnergyShifts = {scatIndex: 0}
    else:
        scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_long]),axis=0)
        scatTypeLabels[scatIndex] = 'Ionized Impurity, Longitudinal'
        scatTypeEnergyShifts[scatIndex] = 0
    ii1 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[ionized_rate_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Ionized Impurity, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    ii2 = scatIndex - 1
    scatIndex += 1

if acoustic_deformation_potential_scattering == 1:
    acoustic_rates_long = AcousticDeformationElasticScatteringRate(energies,mass_longitudinal)
    acoustic_rates_tran = AcousticDeformationElasticScatteringRate(energies,mass_transverse)
    if scatIndex == 1:
        scatteringRateTable = np.array([acoustic_rates_long])
        scatTypeLabels = {scatIndex: 'Acoustic Deformation Potential, Longitudinal'}
        scatTypeEnergyShifts = {scatIndex: 0}
    else:
        scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_long]),axis=0)
        scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, Longitudinal'
        scatTypeEnergyShifts[scatIndex] = 0
    adp1 = scatIndex - 1
    scatIndex += 1
    scatteringRateTable = np.concatenate((scatteringRateTable,[acoustic_rates_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Acoustic Deformation Potential, Transverse'
    scatTypeEnergyShifts[scatIndex] = 0
    adp2 = scatIndex - 1
    scatIndex += 1


if optical_deformation_potential_scattering == 1:
    optical_absorption_rates = OpticalDeformationAbsorptionScatteringRate(energies,mass_average)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_absorption_rates]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Absorption'
    scatTypeEnergyShifts[scatIndex] = optical_energy
    odp1 = scatIndex - 1
    scatIndex += 1

    optical_emission_rates = OpticalDeformationEmissionScatteringRate(energies,mass_average)
    scatteringRateTable = np.concatenate((scatteringRateTable,[optical_emission_rates]),axis=0)
    scatTypeLabels[scatIndex] = 'Optical Deformation Potential, Emission'
    scatTypeEnergyShifts[scatIndex] = -optical_energy
    odp2 = scatIndex - 1
    scatIndex += 1


if intervalley_deformation_potential_scattering == 1:
    rate_gTA_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Dg_TA,Eg_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gTA_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Eg_TA
    ivdp1 = scatIndex - 1
    scatIndex += 1
    rate_gLA_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Dg_LA,Eg_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLA_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Eg_LA
    ivdp2 = scatIndex - 1
    scatIndex += 1
    rate_gLO_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Dg_LO,Eg_LO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLO_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Eg_LO
    ivdp3 = scatIndex - 1
    scatIndex += 1
    rate_fTA_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Df_TA,Ef_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTA_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Ef_TA
    ivdp4 = scatIndex - 1
    scatIndex += 1
    rate_fLA_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Df_LA,Ef_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fLA_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Ef_LA
    ivdp5 = scatIndex - 1
    scatIndex += 1
    rate_fTO_abs_long = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_longitudinal,Df_TO,Ef_TO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTO_abs_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = Ef_TO
    ivdp6 = scatIndex - 1
    scatIndex += 1

    rate_gTA_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Dg_TA,Eg_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gTA_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Eg_TA
    ivdp7 = scatIndex - 1
    scatIndex += 1
    rate_gLA_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Dg_LA,Eg_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLA_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Eg_LA
    ivdp8 = scatIndex - 1
    scatIndex += 1
    rate_gLO_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Dg_LO,Eg_LO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLO_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Eg_LO
    ivdp9 = scatIndex - 1
    scatIndex += 1
    rate_fTA_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Df_TA,Ef_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTA_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Ef_TA
    ivdp10 = scatIndex - 1
    scatIndex += 1
    rate_fLA_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Df_LA,Ef_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fLA_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Ef_LA
    ivdp11 = scatIndex - 1
    scatIndex += 1
    rate_fTO_abs_tran = EquivalentIntervalleyAbsorptionScatteringRate(energies,mass_transverse,Df_TO,Ef_TO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTO_abs_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Absorption, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = Ef_TO
    ivdp12 = scatIndex - 1
    scatIndex += 1

    rate_gTA_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Dg_TA,Eg_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gTA_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Eg_TA
    ivdp13 = scatIndex - 1
    scatIndex += 1
    rate_gLA_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Dg_LA,Eg_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLA_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Eg_LA
    ivdp14 = scatIndex - 1
    scatIndex += 1
    rate_gLO_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Dg_LO,Eg_LO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLO_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Eg_LO
    ivdp15 = scatIndex - 1
    scatIndex += 1
    rate_fTA_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Df_TA,Ef_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTA_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Ef_TA
    ivdp16 = scatIndex - 1
    scatIndex += 1
    rate_fLA_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Df_LA,Ef_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fLA_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Ef_LA
    ivdp17 = scatIndex - 1
    scatIndex += 1
    rate_fTO_em_long = EquivalentIntervalleyEmissionScatteringRate(energies,mass_longitudinal,Df_TO,Ef_TO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTO_em_long]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Longitudinal'
    scatTypeEnergyShifts[scatIndex] = -Ef_TO
    ivdp18 = scatIndex - 1
    scatIndex += 1

    rate_gTA_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Dg_TA,Eg_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gTA_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Eg_TA
    ivdp19 = scatIndex - 1
    scatIndex += 1
    rate_gLA_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Dg_LA,Eg_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLA_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Eg_LA
    ivdp20 = scatIndex - 1
    scatIndex += 1
    rate_gLO_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Dg_LO,Eg_LO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_gLO_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, g-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Eg_LO
    ivdp21 = scatIndex - 1
    scatIndex += 1
    rate_fTA_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Df_TA,Ef_TA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTA_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Ef_TA
    ivdp22 = scatIndex - 1
    scatIndex += 1
    rate_fLA_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Df_LA,Ef_LA)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fLA_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Ef_LA
    ivdp23 = scatIndex - 1
    scatIndex += 1
    rate_fTO_em_tran = EquivalentIntervalleyEmissionScatteringRate(energies,mass_transverse,Df_TO,Ef_TO)
    scatteringRateTable = np.concatenate((scatteringRateTable,[rate_fTO_em_tran]),axis=0)
    scatTypeLabels[scatIndex] = 'Intervalley Potential, Emission, f-type, Transverse'
    scatTypeEnergyShifts[scatIndex] = -Ef_TO
    ivdp24 = scatIndex - 1
    scatIndex += 1

scatteringSums = np.zeros_like(scatteringRateTable)
for i in range(len(scatteringRateTable)):
    if i == 0:
        scatteringSums[i] = scatteringRateTable[i]
    if i > 0:
        scatteringSums[i] = scatteringRateTable[i]+scatteringSums[i-1]
max_scatter = np.max(scatteringSums[-1])


scatTypeConverterLongitudinal = np.array([ii1,adp1,odp1,odp2,ivdp10,ivdp11,ivdp12,ivdp22,ivdp23,ivdp24])
scatteringRateTableLongitudinal = scatteringRateTable[scatTypeConverterLongitudinal]
scatteringRateTableLongitudinal[4:] *= 2 #there are two transverse valleys a longitudinal electron can scatter into, and no longitudinal valleys
scatteringSumsLongitudinal = np.zeros_like(scatteringRateTableLongitudinal)
for i in range(len(scatteringRateTableLongitudinal)):
    if i == 0:
        scatteringSumsLongitudinal[i] = scatteringRateTableLongitudinal[i]
    if i > 0:
        scatteringSumsLongitudinal[i] = scatteringRateTableLongitudinal[i]+scatteringSumsLongitudinal[i-1]
max_scatter_longitudinal = np.max(scatteringSumsLongitudinal[-1])

scatTypeConverterTransverse = np.array([ii2,adp2,odp1,odp2,ivdp1,ivdp2,ivdp3,ivdp4,ivdp5,ivdp6,ivdp13,ivdp14,ivdp15,ivdp16,ivdp17,ivdp18])
scatteringRateTableTransverse = scatteringRateTable[scatTypeConverterTransverse]
scatteringSumsTransverse = np.zeros_like(scatteringRateTableTransverse)
for i in range(len(scatteringRateTableTransverse)):
    if i == 0:
        scatteringSumsTransverse[i] = scatteringRateTableTransverse[i]
    if i > 0:
        scatteringSumsTransverse[i] = scatteringRateTableTransverse[i]+scatteringSumsTransverse[i-1]
max_scatter_transverse = np.max(scatteringSumsTransverse[-1])


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

##### Generate random flight time before collision #####

time_step_count = 80
particle_count = 1000
start_time = 0
max_time = 2e-12 #run the simulation for 5 picoseconds
dt = (max_time-start_time)/time_step_count
times = np.linspace(start_time,max_time,time_step_count)

vxArray = np.random.rand(particle_count)*1e5
vyArray = np.random.rand(particle_count)*1e5
vzArray = np.random.rand(particle_count)*1e5

# electron_states = np.random.rand(particle_count)*2
electron_states = np.zeros(particle_count)
electron_states = electron_states.astype(int)
mass_array = np.array([mass_longitudinal,mass_transverse])
max_scatter_array = np.array([max_scatter_longitudinal,max_scatter_transverse])

energyAveragesLongitudinal = np.zeros([time_step_count])
energyAveragesTransverse = np.zeros([time_step_count])
energyAverages = np.zeros([time_step_count])
velocityAveragesLongitudinal = np.zeros([time_step_count])
velocityAveragesTransverse = np.zeros([time_step_count])
velocityAverages = np.zeros([time_step_count])

electric_field = np.array([0,0,2e6])
scatCount = np.zeros(len(scatteringRateTable))
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
                for scatNum in range(len(scatteringSumsLongitudinal)):
                    if randomScats[i] < scatteringSumsLongitudinal[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        scatteringType[i] = scatTypeConverterLongitudinal[scatNum]+1
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterLongitudinal[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsLongitudinal)-1:
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, Longitudinal':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, f-type, Transverse' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, f-type, Transverse':
                        electron_states[i] = 1
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
                for scatNum in range(len(scatteringSumsTransverse)):
                    if randomScats[i] < scatteringSumsTransverse[scatNum,currentEnergyIndex] and randomScats[i] != 0.0:
                        scatteringType[i] = scatTypeConverterTransverse[scatNum]+1
                        scatYesOrNo[i] = 1
                        scatCount[scatTypeConverterTransverse[scatNum]] += 1
                        break
                    elif scatNum == len(scatteringSumsTransverse)-1:
                        scatYesOrNo[i] = 0
                        scatteringType[i] = 0
                if scatteringType[i] != 0:
                    old_mass = mass_array[electron_states[i]]
                    if scatTypeLabels[scatteringType[i]] == 'Ionized Impurity, Transverse':
                        scatYesOrNo[i] = 2
                        scatterTheta[i] = RandomThetaSelectorIonizedImpurityScattering(currentEnergies[i],mass_array[electron_states[i]])
                    if scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Absorption, f-type, Longitudinal' or scatTypeLabels[scatteringType[i]] == 'Intervalley Potential, Emission, f-type, Longitudinal':
                        electron_states[i] = 0
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
    velocityAveragesLongitudinal[stepNum] = np.mean(vzArray[electron_states == 0])
    if np.isnan(velocityAveragesLongitudinal[stepNum]) == True:
        velocityAveragesLongitudinal[stepNum] = 0
    velocityAveragesTransverse[stepNum] = np.mean(vzArray[electron_states == 1])
    if np.isnan(velocityAveragesTransverse[stepNum]) == True:
        velocityAveragesTransverse[stepNum] = 0
    velocityAverages[stepNum] = np.mean(vzArray)
    energyAveragesLongitudinal[stepNum] = np.mean(energizer(vxArray[electron_states == 0],vyArray[electron_states == 0],vzArray[electron_states == 0],mass_array[electron_states[electron_states == 0]]))
    if np.isnan(energyAveragesLongitudinal[stepNum]) == True:
        energyAveragesLongitudinal[stepNum] = 0
    energyAveragesTransverse[stepNum] = np.mean(energizer(vxArray[electron_states == 1],vyArray[electron_states == 1],vzArray[electron_states == 1],mass_array[electron_states[electron_states == 1]]))
    if np.isnan(energyAveragesTransverse[stepNum]) == True:
        energyAveragesTransverse[stepNum] = 0
    energyAverages[stepNum] = np.mean(energizer(vxArray,vyArray,vzArray,mass_array[electron_states]))
    print(velocityAveragesLongitudinal[stepNum])
    print(velocityAveragesTransverse[stepNum])
    print(velocityAverages[stepNum])

plt.figure(1)
plt.plot(times*1e12,velocityAverages/1e5,'o',label='Overall')
plt.plot(times*1e12,velocityAveragesLongitudinal/1e5,'o',label='Longitudinal')
plt.plot(times*1e12,velocityAveragesTransverse/1e5,'o',label='Transverse')
plt.legend()
plt.figure(2)
plt.plot(times*1e12,energyAverages/1.602e-19,'o',label='Overall')
plt.plot(times*1e12,energyAveragesLongitudinal/e,'o',label='Longitudinal')
plt.plot(times*1e12,energyAveragesTransverse/e,'o',label='Transverse')
plt.legend()
plt.show()

quit
