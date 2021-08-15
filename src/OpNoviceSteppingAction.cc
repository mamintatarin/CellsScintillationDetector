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
//
/// \file OpNoviceSteppingAction.cc
/// \brief Implementation of the OpNoviceSteppingAction class

#include "OpNoviceSteppingAction.hh"
#include <G4SystemOfUnits.hh>
#include "G4Step.hh"
#include "G4OpticalPhoton.hh"
#include "DataFileManager.hh"
#include <cstdlib>

#include <stdio.h>
#include <Logger.hh>
#include"G4Gamma.hh"





OpNoviceSteppingAction::OpNoviceSteppingAction(): G4UserSteppingAction(){}



OpNoviceSteppingAction::~OpNoviceSteppingAction(){}



void OpNoviceSteppingAction::UserSteppingAction(const G4Step* step)
{
  auto track = step->GetTrack();
  /*
    if ((track->GetVolume()->GetName()=="Detector1" || track->GetVolume()->GetName()=="Detector2"  ||
    track->GetVolume()->GetName()=="Detector3" ||track->GetVolume()->GetName()=="Detector4" )

    && track->GetDefinition()==G4OpticalPhoton::Definition()){
          std::printf("%i \n",track->GetTrackID());

        }

 */
    if (track->GetVolume()->GetName()=="Detector1" && track->GetDefinition()== G4OpticalPhoton::Definition()
    &&track->GetVolume()->GetName()!="Scintillator")
    {


            results[0] += track->GetKineticEnergy()/keV;

        track->SetTrackStatus(fStopAndKill);

    }
    if (track->GetVolume()->GetName()=="Detector2" && track->GetDefinition()== G4OpticalPhoton::Definition()
                                                      &&
                                                      track->GetVolume()->GetName()!="Scintillator" ){


            results[1] += track->GetKineticEnergy()/keV;

        track->SetTrackStatus(fStopAndKill);
    }
    if (track->GetVolume()->GetName()=="Detector3" && track->GetDefinition()== G4OpticalPhoton::Definition()
        &&track->GetVolume()->GetName()!="Scintillator")
    {


            results[2] += track->GetKineticEnergy()/keV;

        track->SetTrackStatus(fStopAndKill);
    }
    if (track->GetVolume()->GetName()=="Detector4" && track->GetDefinition()== G4OpticalPhoton::Definition()
    &&track->GetVolume()->GetName()!="Scintillator")
    {


            results[3] += track->GetKineticEnergy()/keV;

        track->SetTrackStatus(fStopAndKill);
    }




/*

    if (track->GetVolume()->GetName()=="Scintillator" && track->GetDefinition()==G4OpticalPhoton::Definition())
    {


        results[0] += track->GetKineticEnergy()/keV;
        track->SetTrackStatus(fStopAndKill);
    }
 */

    }


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
