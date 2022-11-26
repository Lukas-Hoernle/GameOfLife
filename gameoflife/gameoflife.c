#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define calcIndex(width, x,y)  ((y)*(width) + (x))
long TimeSteps = 100;

void writeVTK2(long timestep, double *data, char prefix[1024], int w, int h) {
  char filename[2048];  
  int x,y; 
  
  int offsetX=0;
  int offsetY=0;
  float deltax=1.0;
  long  nxy = w * h * sizeof(float);  

  snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".vti");
  FILE* fp = fopen(filename, "w");

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX, offsetX + w, offsetY, offsetY + h, 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
  fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</ImageData>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fwrite((unsigned char*)&nxy, sizeof(long), 1, fp);

  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      float value = data[calcIndex(h, x,y)];
      fwrite((unsigned char*)&value, sizeof(float), 1, fp);
    }
  }
  
  fprintf(fp, "\n</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}


void show(double* currentfield, int w, int h) {
  printf("\033[H");
  int x,y;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) printf(currentfield[calcIndex(w, x,y)] ? "\033[07m  \033[m" : "  ");
    //printf("\033[E");
    printf("\n");
  }
  fflush(stdout);
}

int calculateNeighbours(int x, int y,int w,int h,double* currentfield){
  int neighboursAlive=0;
  for(int neighbourX=-1;neighbourX<2;++neighbourX){
      for(int neighbourY=-1;neighbourY<2;++neighbourY){
        if(neighbourX == 0 && neighbourY == 0){
          continue;
        }
        if(x % w ==1){//correct y if x overflow 
          y--;
        }
        if(y%h==1){//correct x if y overflow 
          x--;
        }
        neighboursAlive += currentfield[calcIndex(w,x + neighbourX, y + neighbourY)];        
      } 
    }
    return neighboursAlive;
} 

void evolve(double* currentfield, double* newfield, int w, int h,int xThreadNum,int yThreadNum){
  int x=0,y=0,neighboursAlive=0;
  #pragma omp parallel firstprivate(currentfield,x,y,neighboursAlive) shared(newfield,w,h) num_threads(5) 
  {
   int xStart = omp_get_thread_num() % w;
   int yStart = omp_get_thread_num() / x;
   int xEnd = omp_get_thread_num() % w + (w/xThreadNum);
   int yEnd = omp_get_thread_num() / x + (h/yThreadNum);
  for(x=xStart;x<=xEnd;x++){
    for(y=yStart;y<=yEnd;y++){
      neighboursAlive=calculateNeighbours(x,y,w,h,currentfield);
      switch(neighboursAlive){
        case(3): newfield[calcIndex(w,x,y)]=1;break;
        case(2): newfield[calcIndex(w,x,y)]=currentfield[calcIndex(w,x,y)];break;
        default: newfield[calcIndex(w,x,y)]=0;break;
      }
    }
  }
  }
}


 
void filling(double* currentfield, int w, int h) {
  int i;
  for (i = 0; i < h*w; i++) {
    currentfield[i] = (rand() < RAND_MAX / 10) ? 1 : 0; ///< init domain randomly
  }
}
 



void game(int w, int h,int xThreadNum,int yThreadNum) {
  double *currentfield = calloc(w*h, sizeof(double));
  double *newfield     = calloc(w*h, sizeof(double));
  filling(currentfield, w, h);
  long t;
  for (t=0;t<TimeSteps;t++) {
    show(currentfield, w, h);
    evolve(currentfield, newfield, w, h,xThreadNum,yThreadNum);
    
    printf("%ld timestep\n",t);
    writeVTK2(t,currentfield,"gol", w, h);
    
    usleep(200000);

    //SWAP
    double *temp = currentfield;
    currentfield = newfield;
    newfield = temp;
  }
  free(currentfield);
  free(newfield);
  
}
 
int main(int c, char **v) {
  int xThreadNum=1,yThreadNum=1,xThreadSize=1,yThreadSize=1;
  if (c > 1) xThreadNum = atoi(v[1]); ///< read 
  if (c > 2) yThreadNum = atoi(v[2]); ///< read 
  if (c > 3) xThreadSize = atoi(v[3]); ///< read //die Dinger übergeben dann in der for schleife festlegen, dass von xstart bis xende geht und die aus diesen dingens berechnen
  if (c > 4) yThreadSize = atoi(v[4]); ///< read 
  int w = xThreadNum*xThreadSize;
  int h =yThreadNum*yThreadSize;
  game(w, h,xThreadNum,yThreadNum);
}
