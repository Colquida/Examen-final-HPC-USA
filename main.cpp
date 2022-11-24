/* Fecha: 21 sept 2022
 * Autor:  Jonathan Alexander Torres Benítez
 * Materia: HPC
 * Tópico : Implementación de las Regresión Lineal como modelo en C++
 * Requerimientos:
 *  - Constriur una clase Extracción, que permita
 *  manipular, extraer y cargar los datos.
 *  - Construir una clase LinearRegression, que permita
 *  los calculos de la función de costo, gradientes descendiente
 *  entre otras
 *
 * */

#include "ClassExtraction/extractiondata.h"
#include "Regression/regression.h"
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <list>
#include <vector>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[])
{

    //Se crea un objeto del tipo ClassExtraction
    ExtractionData ExData(argv[1],argv[2],argv[3]);
    //SE instancia la clase de regresión lineal en un objeto

    Regression modeloLR;

    //Se crea un vector de vectores del tipo string para cargar objeto ExData
    std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();

    //Cantidad de filas y columnas
    int filas    = dataframe.size();
    int columnas = dataframe[0].size();

    //Se crea una matriz Eigen, para ingresar los valores a esa matriz
    Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);


    /*Se normaliza la matriz de los datos */
    Eigen::MatrixXd matNorm = ExData.Norm(matData);


    /*Se divide en datos de entrenamiento y datos de prueba*/
    Eigen::MatrixXd X_train, y_train, X_test, y_test;


    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> tupla_datos = ExData.TrainTestSplit(matNorm, 0.8);
    /*Se descomprime la tupla en 4 conjuntos */

    std::tie(X_train,y_train,X_test,y_test) = tupla_datos;

    /* Se crea vectores auxiliares para prueba y entrenamiento inicializados en 1 */
    Eigen::VectorXd vector_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vector_test = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensiona la matriz de entrenamiento y de prueba para ser ajustadas a
     * los vectores auxiliares anteriores */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vector_train;
    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vector_test;

    /* Se crea el vector de coeficientes theta */
    Eigen::VectorXd thetas =Eigen::VectorXd::Zero(X_train.cols());
    /* Se establece el alpha como ratio de aprendizaje de tipo flotante */
    float alpha = 0.01;
    int num_iter = 1000; //num iteraciones

    //Se crea un vector para almacenar las thetas de salida (parametro)
    Eigen::MatrixXd thetas_out;
    //Se crea un vector sencillo(std) de flotantes para almacenar los valores del costo
    std::vector<float> costo;
    //Se calcula el gradeinte descendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> g_descendiente = modeloLR.GradienteDescent(X_train,
                                                                                               y_train,
                                                                                               thetas,
                                                                                               alpha,
                                                                                               num_iter);

    //Se desempaqueta el gradiente
    std::tie(thetas_out,costo) = g_descendiente;

     /*SE almacena los valores de thetas y costos en un fichero para psoteriormente ser visualizados */

    //ExData.VectortoFile(costo, "costo.txt");
    //ExData.EigentoFile(thetas_out, "thetas.txt");

    /*SE extrae el promedio de la matriz de entrada */

    auto prom_data = ExData.Promedio(matData);
    //Se extraen los valores de la varibales independientes
    //auto var_prom_independientes = prom_data(0,11); <- original
    auto var_prom_independientes = prom_data(0,10);
    //SE escalan los datos
    auto datos_escalados = matData.rowwise() - matData.colwise().mean();
    //Se extrae la desviación estandar de datos escalados
    auto dev_stand = ExData.DevStand(datos_escalados);
    //Se extrane los valores de las viarbales independientes de la desviación estandar
    //auto var_des_independientes =dev_stand(0,11);
    auto var_des_independientes =dev_stand(0,10);
    //Se crea una maitriz para almacenar los valores estimados de entrenamiento.
    Eigen::MatrixXd y_train_hat = (X_train * thetas_out * var_des_independientes).array() + var_prom_independientes;
    //Matriz para los valores reales de y
    //Eigen::MatrixXd y = matData.col(11).topRows(1278);
    Eigen::MatrixXd y = matData.col(10).topRows(545);

    //Eigen::MatrixXd y_test_b = matData.col(10).bottomRows(136);

    //Eigen::MatrixXd y_test_hat = (X_test * thetas_out * var_des_independientes).array() + var_prom_independientes;

    //Se revisa que tan bueno quedó el modelo a traves de una metrica de rendimiento
    float metrica_R2 = modeloLR.r2_score(y,y_train_hat);
    //float metrica_R2_y = modeloLR.r2_score(y_test_b,y_test_hat);


    std::cout << "metrica_R2: " <<metrica_R2 <<std::endl;
    //std::cout << "metrica_R2: " <<metrica_R2_y <<std::endl;

    float metrica_MSE = modeloLR.mse(y, y_train_hat);
    std::cout<< "Métrica MSE "<<metrica_MSE<<std::endl;
    float metrica_RMSE = modeloLR.rmse(y, y_train_hat);
    std::cout<< "Métrica RMSE "<<metrica_RMSE<<std::endl;



    std::cout<<"Matriz de entrenamiento, número " << X_train.rows() << " de Filas"<<std::endl;
    std::cout<<"Matriz de prueba, número " << X_test.rows() << " de Filas"<<std::endl;
    std::cout<<"Matriz total, número " << matNorm.rows() << " de Filas"<<std::endl;
    std::cout<<"El promedio de la matriz es de: "<<ExData.Promedio(matData)<<"\n"<<std::endl;
    std::cout<<"La desviación de los valores en cada columna es de: "<<ExData.DevStand(datos_escalados)<<"\n"<<std::endl;

    return EXIT_SUCCESS;
}
