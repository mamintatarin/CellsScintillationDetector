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
/// \file OpNovice/src/OpNoviceDetectorConstruction.cc
/// \brief Implementation of the OpNoviceDetectorConstruction class
//
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "OpNoviceDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include <Logger.hh>
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNoviceDetectorConstruction::OpNoviceDetectorConstruction()
 : G4VUserDetectorConstruction()
{
  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNoviceDetectorConstruction::~OpNoviceDetectorConstruction(){;}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* OpNoviceDetectorConstruction::Construct()
{

  G4String name, symbol;

  // Get nist material manager
  G4NistManager* nist = G4NistManager::Instance();
  

  G4bool isotopes = true;
  


    G4Material* GAGG = new G4Material("GAGG", 6.63*g/cm3, 4);
    G4Element*  Al = nist->FindOrBuildElement("Al" , isotopes);
    G4Element* O = nist->FindOrBuildElement("O", isotopes);
    G4Element* Gd = nist->FindOrBuildElement("Gd", isotopes);
    G4Element* Ga = nist->FindOrBuildElement("Ga", isotopes);
    GAGG->AddElement(Al,2);
    GAGG->AddElement(Gd,3);
    GAGG->AddElement(Ga,3);
    GAGG->AddElement(O,12);



  // Option to switch on/off checking of volumes overlaps
  //
  G4bool checkOverlaps = true;
    
    
    G4int n = 2;
    G4double ph_Energy[]    = { 1.0*eV , 7.07*eV};
    G4double LaBr3RefractiveIndex[]= {1.90,1.90};
    G4double LaBr3Absorlen[]= {100*cm,100*cm};

    

    G4MaterialPropertiesTable* GAGGSC = new G4MaterialPropertiesTable();


    //G4double distrEn[]={2.6*eV,2.38*eV,2.07*eV,1.77*eV};
    //G4double distrF[]={0.1,1,0.33,0.08};
    G4double distrEn[]={2.38*eV,2.4*eV};
    G4double distrF[]={0.5,0.5};
    G4double resolution_factor=3.59;
    GAGGSC->AddConstProperty("SCINTILLATIONYIELD",40./keV);
    GAGGSC->AddProperty("RINDEX", ph_Energy,LaBr3RefractiveIndex,n);
    GAGGSC->AddProperty("ABSLENGTH",ph_Energy, LaBr3Absorlen,n);
    GAGGSC->AddConstProperty("RESOLUTIONSCALE",resolution_factor);
    GAGGSC->AddConstProperty("YIELDRATIO",0.95);
    GAGGSC->AddConstProperty("FASTTIMECONSTANT", 5*ns);
    GAGGSC->AddConstProperty("SLOWTIMECONSTANT", 60*ns);
    //GAGGSC->AddProperty("FASTCOMPONENT",distrEn,distrF,4)->SetSpline(TRUE);
    //GAGGSC->AddProperty("SLOWCOMPONENT",distrEn,distrF,4)->SetSpline(TRUE);
    GAGGSC->AddProperty("FASTCOMPONENT",distrEn,distrF,2);
    GAGGSC->AddProperty("SLOWCOMPONENT",distrEn,distrF,2);
    
    Logger::instance()->print(("RESOLUTIONSCALE: "+std::to_string(resolution_factor)).c_str());
    GAGG->SetMaterialPropertiesTable(GAGGSC);
    
    

  //
  // World
  //
  G4double world_sizeXY = 30*cm;
  G4double world_sizeZ  = 30*cm;
  G4Material* world_mat = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* det_mat = nist->FindOrBuildMaterial("G4_AIR");
  //G4Material* muscle_mat = nist->FindOrBuildMaterial("G4_MUSCLE_SKELETAL_ICRP");
  G4MaterialPropertiesTable* GalacticSC = new G4MaterialPropertiesTable();
  G4MaterialPropertiesTable* detTable = new G4MaterialPropertiesTable();
  G4double refr[]={1.9,1.9};
  G4double refrDet[]={1.9,1.9};
  GalacticSC->AddProperty("RINDEX",ph_Energy,refr,2);
  detTable->AddProperty("RINDEX",ph_Energy,refrDet,2);
  world_mat ->SetMaterialPropertiesTable(GalacticSC);
  det_mat->SetMaterialPropertiesTable(detTable);






  
  G4Box* solidWorld =
    new G4Box("World",                       //its name
       world_sizeXY, world_sizeXY, world_sizeZ);     //its size
      
  G4LogicalVolume* logicWorld =
    new G4LogicalVolume(solidWorld,          //its solid
                        world_mat,           //its material
                        "World");            //its name
                                   
  G4VPhysicalVolume* physWorld =
    new G4PVPlacement(0,                     //no rotation
                      G4ThreeVector(),       //at (0,0,0)
                      logicWorld,            //its logical volume
                      "World",               //its name
                      0,                     //its mother  volume
                      false,                 //no boolean operation
                      0,                     //copy number
                      checkOverlaps);        //overlaps checking
                     

   //
  // Scintillator
  //
  G4double scintillator_sizeXY = 0.7*cm;
  G4double scintillator_sizeZ  = 0.35*cm;
  
  
  G4Box* solidScintillator =
    new G4Box("Scintillator",                       //its name
       scintillator_sizeXY, scintillator_sizeXY, scintillator_sizeZ);     //its size
      
  G4LogicalVolume* logicScintillator =
    new G4LogicalVolume(solidScintillator,          //its solid
                        GAGG,           //its material
                        "Scintillator");            //its name
                                   
  G4VPhysicalVolume* Scintillator =
    new G4PVPlacement(0,                     //no rotation
                      G4ThreeVector(),       //at (0,0,0)
                      logicScintillator,            //its logical volume
                      "Scintillator",               //its name
                      logicWorld,                     //its mother  volume
                      false,                 //no boolean operation
                      0,                     //copy number
                      checkOverlaps);        //overlaps checking
 //meat

    G4Box* solidAirBox =
            new G4Box("airbox",                       //its name
                      scintillator_sizeXY, scintillator_sizeXY, 0.5*mm);     //its size

    /* G4Box* solidMuscle =
          new G4Box("muscle",                       //its name
                    3.5*cm, 3.5*cm, 5*cm);     //its size
  G4LogicalVolume* logicMuscle =
          new G4LogicalVolume(solidMuscle,          //its solid
                              muscle_mat,           //its material
                              "muscle");            //its name
  G4VPhysicalVolume* muscle =
          new G4PVPlacement(0,                     //no rotation
                            G4ThreeVector(0,0,scintillator_sizeZ+5*cm+1*mm),
                            logicMuscle,            //its logical volume
                            "muscle",               //its name
                            logicWorld,                     //its mother  volume
                            false,                 //no boolean operation
                            0,                     //copy number
                            checkOverlaps);        //overlaps checking
                             */
    //side borders


    G4LogicalVolume* logicDegree1 =
            new G4LogicalVolume(solidScintillator,          //its solid
                                world_mat,           //its material
                                "Degree1");            //its name

    G4VPhysicalVolume* Degree1 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(1.4*cm,0.0*cm,0.0*cm),
                              logicDegree1,            //its logical volume
                              "Degree1",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking

    G4LogicalVolume* logicDegree2 =
            new G4LogicalVolume(solidScintillator,          //its solid
                                world_mat,           //its material
                                "Degree2");            //its name

    G4VPhysicalVolume* Degree2 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(-1.4*cm,0.0*cm,0.0*cm),
                              logicDegree2,            //its logical volume
                              "Degree2",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking
    G4LogicalVolume* logicDegree3 =
            new G4LogicalVolume(solidScintillator,          //its solid
                                world_mat,           //its material
                                "Degree3");            //its name

    G4VPhysicalVolume* Degree3 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(0*cm,1.4*cm,0.0*cm),
                              logicDegree3,            //its logical volume
                              "Degree3",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking

    G4LogicalVolume* logicDegree4 =
            new G4LogicalVolume(solidScintillator,          //its solid
                                world_mat,           //its material
                                "Degree4");            //its name

    G4VPhysicalVolume* Degree4 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(0*cm,-1.4*cm,0.0*cm),       //at (0,0,0)
                              logicDegree4,            //its logical volume
                              "Degree4",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking
    G4LogicalVolume* logicDegree5 =
            new G4LogicalVolume(solidAirBox,          //its solid
                                world_mat,           //its material
                                "Degree5");            //its name

    G4VPhysicalVolume* Degree5 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(0*cm,0*cm,scintillator_sizeZ+0.5*mm),       //at (0,0,0)
                              logicDegree5,            //its logical volume
                              "Degree5",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking



     //downside detectors

    G4Box* solidDetector =
            new G4Box("Detector",                       //its name
                      scintillator_sizeXY/2, scintillator_sizeXY/2, scintillator_sizeZ);     //its size
    G4LogicalVolume* logicDetector1 =
            new G4LogicalVolume(solidDetector,          //its solid
                                det_mat,           //its material
                                "Detector1");            //its name

    G4VPhysicalVolume* Detector1 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(scintillator_sizeXY/2,scintillator_sizeXY/2,-2*scintillator_sizeZ),       //at (0,0,0)
                              logicDetector1,            //its logical volume
                              "Detector1",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking
    G4LogicalVolume* logicDetector2 =
            new G4LogicalVolume(solidDetector,          //its solid
                                det_mat,           //its material
                                "Detector2");            //its name

    G4VPhysicalVolume* Detector2 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(-scintillator_sizeXY/2,scintillator_sizeXY/2,-2*scintillator_sizeZ),       //at (0,0,0)
                              logicDetector2,            //its logical volume
                              "Detector2",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking

    G4LogicalVolume* logicDetector3 =
            new G4LogicalVolume(solidDetector,          //its solid
                                det_mat,           //its material
                                "Detector3");            //its name

    G4VPhysicalVolume* Detector3 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(-scintillator_sizeXY/2,-scintillator_sizeXY/2,-2*scintillator_sizeZ),       //at (0,0,0)
                              logicDetector3,            //its logical volume
                              "Detector3",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking

    G4LogicalVolume* logicDetector4 =
            new G4LogicalVolume(solidDetector,          //its solid
                                det_mat,           //its material
                                "Detector4");            //its name

    G4VPhysicalVolume* Detector4 =
            new G4PVPlacement(0,                     //no rotation
                              G4ThreeVector(scintillator_sizeXY/2,-scintillator_sizeXY/2,-2*scintillator_sizeZ),       //at (0,0,0)
                              logicDetector4,            //its logical volume
                              "Detector4",               //its name
                              logicWorld,                     //its mother  volume
                              false,                 //no boolean operation
                              0,                     //copy number
                              checkOverlaps);        //overlaps checking

    //surfaces

    G4OpticalSurface* OpSurface1 = new G4OpticalSurface("side1");
    G4LogicalBorderSurface* Surface1 = new
            G4LogicalBorderSurface("side1",Scintillator,Degree1,OpSurface1);

    G4OpticalSurface* OpSurface2 = new G4OpticalSurface("side2");
    G4LogicalBorderSurface* Surface2 = new
            G4LogicalBorderSurface("side2",Scintillator,Degree2,OpSurface2);

    G4OpticalSurface* OpSurface3 = new G4OpticalSurface("side3");
    G4LogicalBorderSurface* Surface3 = new
            G4LogicalBorderSurface("side3",Scintillator,Degree3,OpSurface3);

    G4OpticalSurface* OpSurface4 = new G4OpticalSurface("side4");
    G4LogicalBorderSurface* Surface4 = new
            G4LogicalBorderSurface("side4",Scintillator,Degree4,OpSurface4);

    G4OpticalSurface* OpSurface5 = new G4OpticalSurface("side5");
    G4LogicalBorderSurface* Surface5 = new
            G4LogicalBorderSurface("side5",Scintillator,Degree5,OpSurface5);


    G4OpticalSurfaceFinish finishSide=polishedfrontpainted;
    G4OpticalSurfaceModel modelSide=unified;
    G4SurfaceType typeSide=dielectric_dielectric;
	
	G4OpticalSurfaceFinish finishRoof=polishedfrontpainted;
    G4OpticalSurfaceModel modelRoof=unified;
    G4SurfaceType typeRoof=dielectric_dielectric;
	/*
    G4OpticalSurfaceFinish finishRoof=etchedtyvekair;;
    G4OpticalSurfaceModel modelRoof=LUT;
    G4SurfaceType typeRoof=dielectric_LUT;

    G4OpticalSurfaceFinish finishSide=polishedtyvekair;
    G4OpticalSurfaceModel modelSide=LUT;
    G4SurfaceType typeSide=dielectric_LUT;
    */
    Logger::instance()->print(("Surface type (roof): "+std::to_string(typeRoof)).c_str());
    Logger::instance()->print(("Surface model (roof): "+std::to_string(modelRoof)).c_str());
    Logger::instance()->print(("Surface finish (roof): "+std::to_string(finishRoof)).c_str());
    Logger::instance()->print(("Surface type (side): "+std::to_string(typeSide)).c_str());
    Logger::instance()->print(("Surface model (side): "+std::to_string(modelSide)).c_str());
    Logger::instance()->print(("Surface finish (side): "+std::to_string(finishSide)).c_str());

    OpSurface1->SetType(typeSide);
    OpSurface1->SetFinish(finishSide);
    OpSurface1->SetModel(modelSide);
    OpSurface2->SetFinish(finishSide);
    OpSurface2->SetModel(modelSide);
    OpSurface2->SetType(typeSide);
    OpSurface3->SetFinish(finishSide);
    OpSurface3->SetModel(modelSide);
    OpSurface3->SetType(typeSide);
    OpSurface4->SetFinish(finishSide);
    OpSurface4->SetModel(modelSide);
    OpSurface4->SetType(typeSide);

    OpSurface5->SetFinish(finishRoof);
    OpSurface5->SetModel(modelRoof);
    OpSurface5->SetType(typeRoof);




    G4double reflectivitySideOp[]={this->side,this->side};
    G4double reflectivityRoofOp[]={this->roof,this->roof};

   
    G4MaterialPropertiesTable* OpSurfaceProperty1 = new G4MaterialPropertiesTable();
    OpSurfaceProperty1->AddProperty("REFLECTIVITY",ph_Energy,reflectivitySideOp,n);
    G4MaterialPropertiesTable* OpSurfaceProperty2 = new G4MaterialPropertiesTable();
    OpSurfaceProperty2->AddProperty("REFLECTIVITY",ph_Energy,reflectivitySideOp,n);
    G4MaterialPropertiesTable* OpSurfaceProperty3 = new G4MaterialPropertiesTable();
    OpSurfaceProperty3->AddProperty("REFLECTIVITY",ph_Energy,reflectivitySideOp,n);
    G4MaterialPropertiesTable* OpSurfaceProperty4 = new G4MaterialPropertiesTable();
    OpSurfaceProperty4->AddProperty("REFLECTIVITY",ph_Energy,reflectivitySideOp,n);
    G4MaterialPropertiesTable* OpSurfaceProperty5 = new G4MaterialPropertiesTable();
    OpSurfaceProperty5->AddProperty("REFLECTIVITY",ph_Energy,reflectivityRoofOp,n);


    OpSurface1->SetMaterialPropertiesTable(OpSurfaceProperty1);
    OpSurface2->SetMaterialPropertiesTable(OpSurfaceProperty2);
    OpSurface3->SetMaterialPropertiesTable(OpSurfaceProperty3);
    OpSurface4->SetMaterialPropertiesTable(OpSurfaceProperty4);
    OpSurface5->SetMaterialPropertiesTable(OpSurfaceProperty5);


    G4OpticalSurface* OpSurface6 = new G4OpticalSurface("detector1");
    G4LogicalBorderSurface* Surface6 = new
            G4LogicalBorderSurface("detector1",Scintillator,Detector1,OpSurface6);
    G4OpticalSurface* OpSurface7 = new G4OpticalSurface("detector2");
    G4LogicalBorderSurface* Surface7 = new
            G4LogicalBorderSurface("detector2",Scintillator,Detector2,OpSurface7);
    G4OpticalSurface* OpSurface8 = new G4OpticalSurface("detector3");
    G4LogicalBorderSurface* Surface8 = new
            G4LogicalBorderSurface("detector3",Scintillator,Detector3,OpSurface8);
    G4OpticalSurface* OpSurface9 = new G4OpticalSurface("detector4");
    G4LogicalBorderSurface* Surface9= new
            G4LogicalBorderSurface("detector4",Scintillator,Detector4,OpSurface9);
  
    G4OpticalSurfaceFinish finishDet=polished;
    G4OpticalSurfaceModel modelDet=unified;
    G4SurfaceType typeDet=dielectric_dielectric;
    OpSurface6->SetType(typeDet);
    OpSurface6->SetFinish(finishDet);
    OpSurface6->SetModel(modelDet);
    OpSurface7->SetFinish(finishDet);
    OpSurface7->SetModel(modelDet);
    OpSurface7->SetType(typeDet);
    OpSurface8->SetFinish(finishDet);
    OpSurface8->SetModel(modelDet);
    OpSurface8->SetType(typeDet);
    OpSurface9->SetFinish(finishDet);
    OpSurface9->SetModel(modelDet);
    OpSurface9->SetType(typeDet);
    G4double reflectivityDet[]={0.3,0.3};

 

    G4MaterialPropertiesTable* OpSurfaceProperty6 = new G4MaterialPropertiesTable();
    OpSurfaceProperty6->AddProperty("REFLECTIVITY",ph_Energy,reflectivityDet,n);
    G4MaterialPropertiesTable* OpSurfaceProperty7 = new G4MaterialPropertiesTable();
    OpSurfaceProperty7->AddProperty("REFLECTIVITY",ph_Energy,reflectivityDet,n);
    G4MaterialPropertiesTable* OpSurfaceProperty8 = new G4MaterialPropertiesTable();
    OpSurfaceProperty8->AddProperty("REFLECTIVITY",ph_Energy,reflectivityDet,n);
    G4MaterialPropertiesTable* OpSurfaceProperty9 = new G4MaterialPropertiesTable();
    OpSurfaceProperty9->AddProperty("REFLECTIVITY",ph_Energy,reflectivityDet,n);


    OpSurface6->SetMaterialPropertiesTable(OpSurfaceProperty6);
    OpSurface7->SetMaterialPropertiesTable(OpSurfaceProperty7);
    OpSurface8->SetMaterialPropertiesTable(OpSurfaceProperty8);
    OpSurface9->SetMaterialPropertiesTable(OpSurfaceProperty9);

 





    return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
