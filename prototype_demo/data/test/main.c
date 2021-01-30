#include <stdio.h>
#include <stdlib.h>

double
agent(double x, double var)
{

    if (x < var){
        x = x * x;
    }
    else {
        x = 2.0 * x;      
    }
    x += 1.0;
    
    return x;
}

int main(int argc, char **argv)
{
    double thres = 5.0;
    double x = 4.0;
    double result = agent(x, thres);
    
    return 1;
}

