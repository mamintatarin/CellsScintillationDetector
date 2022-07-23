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
/// \file OpNoviceActionInitialization.cc
/// \brief Implementation of the OpNoviceActionInitialization class

#include "OpNoviceActionInitialization.hh"
#include "OpNovicePrimaryGeneratorAction.hh"
#include "OpNoviceRunAction.hh"
#include "OpNoviceSteppingAction.hh"
#include "OpNoviceStackingAction.hh"
#include "OpNoviceEventAction.hh"
#include "Logger.hh"
#include <G4SystemOfUnits.hh>
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNoviceActionInitialization::OpNoviceActionInitialization(CLHEP::Hep3Vector pos)
 : G4VUserActionInitialization()
{
    position=pos;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNoviceActionInitialization::~OpNoviceActionInitialization(){}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void OpNoviceActionInitialization::BuildForMaster() const
{
  SetUserAction(new OpNoviceRunAction(position));
    Logger::instance()->print("BuildForMaster() IS INVOKED!");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void OpNoviceActionInitialization::Build() const
{
    OpNoviceEventAction* tempEvent = new OpNoviceEventAction(x,y);
    OpNoviceSteppingAction* tempStepping =  new OpNoviceSteppingAction();

  OpNoviceRunAction * tempRun = new OpNoviceRunAction(position);
  Logger::instance()->print(std::to_string(position.getX()*mm)+ " " + std::to_string(position.getY()*mm)+" " + std::to_string(position.getZ()*mm) + " mm");

 // tempStepping->eventAct=tempEvent;
  tempEvent->position=position;
  tempEvent->tempStepping=tempStepping;
  SetUserAction(new OpNovicePrimaryGeneratorAction(position));
  SetUserAction(tempRun);
  SetUserAction(tempEvent);
  SetUserAction(tempStepping);
  SetUserAction(new OpNoviceStackingAction());
}
