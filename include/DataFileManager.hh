//
// Created by mihail on 20.04.17.
//

#ifndef GEANT4_THUNDERSTORM_DATAFILEMANAGER_HH
#define GEANT4_THUNDERSTORM_DATAFILEMANAGER_HH

#include <map>
#include <utility>
#include <string>
#include <sys/stat.h>
#include "DataFile.hh"
#include "fileUtils.hh"

using namespace std;

class DataFileManager {
    
public:
    static DataFileManager* instance(){
        static DataFileManager dataFileManager;
        return &dataFileManager;
    }
    template <class Data>
    DataFile<Data> * getDataFile(string name){

        if (dataFileMap.find(name)==dataFileMap.end()){
            dataFileMap[name] = new DataFile<Data>(name);
        }

        return (DataFile<Data>*) dataFileMap[name];

        //return new DataFile<Data>(name);
    }
    ofstream* getTextFile(string name){
        if (textFileMap.find(name)==textFileMap.end()){
            string nameFile = checkFileName(name, 0, ".txt");
            ofstream* fout = new ofstream;
            fout->open("data/" + nameFile + ".txt");
            textFileMap[name] = fout;
        }
        return textFileMap[name];
    }

    ~DataFileManager(){
        for (auto it = dataFileMap.begin(); it != dataFileMap.end(); ++it)
        {
            delete it->second;
        }
    };
private:
    map<string, IDataFile*> dataFileMap;
    map<string, ofstream*> textFileMap;
    DataFileManager(){
        struct stat sb;
        char *dirName = (char *) "data";
        if (stat(dirName, &sb) == 0 && S_ISDIR(sb.st_mode))
        {
//            cout<<"Directory for data files exist"<<endl;
        }
        else
        {
            system("mkdir data");
//            cout<<"Directory for data files created"<<endl;

        }

    };
    DataFileManager(DataFileManager const&);
    DataFileManager& operator=(DataFileManager const&);

};




#endif //GEANT4_THUNDERSTORM_DATAFILEMANAGER_HH
