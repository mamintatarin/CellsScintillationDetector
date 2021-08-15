//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file OpNovice/src/OpNovicePrimaryGeneratorAction.cc
/// \brief Implementation of the OpNovicePrimaryGeneratorAction class
//
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "OpNovicePrimaryGeneratorAction.hh"



#include <Logger.hh>
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include <G4RandomDirection.hh>


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNovicePrimaryGeneratorAction::OpNovicePrimaryGeneratorAction(CLHEP::Hep3Vector pos)
 : G4VUserPrimaryGeneratorAction(), 
   fParticleGun(0)
{
    position=pos;
  G4int n_particle = 1;
  fParticleGun = new G4ParticleGun(n_particle);
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();

        G4double energy=140;
        string particle_type="gamma";

        G4ParticleDefinition* particle = particleTable->FindParticle(particle_type);
         fParticleGun->SetParticleDefinition(particle);
         fParticleGun->SetParticleTime(0.0*ns);
         fParticleGun->SetParticlePosition(position);
         fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,-1.));
         fParticleGun->SetParticleEnergy(energy*keV);
         /*

                 G4double energy=2.4*0.001;
                string particle_type="opticalphoton";
                G4ParticleDefinition* particle = particleTable->FindParticle(particle_type);
                fParticleGun->SetParticleDefinition(particle);
                fParticleGun->SetParticleTime(0.0*ns);
                fParticleGun->SetParticlePosition(G4ThreeVector(0.0,0.0,0.0));
                fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.1,0.1,0.7));
                fParticleGun->SetParticleEnergy(energy*eV);
                fParticleGun->SetParticlePolarization(G4ThreeVector(1.0,1.0,0.0));
        */



     Logger::instance()->print(("Primary particle type: "+particle_type).c_str());
     Logger::instance()->print(("Primary particle energy: "+std::to_string(energy)+" keV").c_str());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNovicePrimaryGeneratorAction::~OpNovicePrimaryGeneratorAction()
{
  delete fParticleGun;
  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void OpNovicePrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
   // fParticleGun->SetParticleMomentumDirection(G4RandomDirection());
    fParticleGun->GeneratePrimaryVertex(anEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

