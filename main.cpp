//g++ amosa_clustering_vga_sym_min.cpp -o amosa -Iinclude -Llib -lANN
/*Multiobjective Euclidean distance Based clustering using FCM, XB and PBM index together */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>
#include<string.h>
//#include "ANN/ANN.h"

#define DIMENSION 900 //may get changed
#define randm(num) 	(rand()%(num))
#define DUMMY 99999.99
#define DUMMY1 99999
#define MIN_LEN 2
#define MAX_LEN 50
#define MAX_GEN  100
#define MAX 5
#define MINFS 25000.0
#define hardl 20
#define softl 30
#define PMUT 0.8
#define NO_OF_ELM 10000
#define MUTATE_STRING_NORMAL 0.4
#define MUTATE_LAPLACIAN 1.0
#define MUTATE_NORMAL 0.75
#define MUTATE_DELETE 0.05
#define MUTATE_INSERT 0.2
#define irand() rand()%1000
#define frand() irand()/1000.0
#define Max(a,b) a>b?a:b
#define Min(a,b) a<b?a:b
#define size 800

struct pattern
{
double x[DIMENSION];
};

//int	nPts;					        // actual number of data points
//ANNpointArray	dataPts;				// data points
//ANNpoint	queryPt;				// query point
//ANNidxArray	nnIdx;					// near neighbor indices
//ANNdistArray	dists;					// near neighbor distances
//ANNkd_tree*	kdTree;
//int knear=2;
double	eps= 0.00;			// error bound
int maxPts= 3000;
double epsilon=0.00;
double threshold=0.8;

int n,d,r,K,no_of_class,no_testpoint,total_pt; //n: number of points to be clustered, d: dimension of each point
int arcsize;
double Temperature, TMin = 0.01, TMax = 100, alpha = 0.8; //TMin: minimum temparature, TMax: Maximum temparature, alpha: cooling rate : User can change the cooling rate between 0.8 to 0.95
int  MaxLen, MinLen;
double **area;
int POP_SIZE;
char file[30];
double *Obj;
double *Obj_current;
int total_index;
int *FitnessIndex;
double **points, **testpoints;
int *trucls;
double *min,*max;
FILE *fpo,*testp;
int *fclass, *testclass;
struct archive_element
{
int len;
struct pattern z[MAX_LEN];
int index1[MAX_LEN];
double **Membership;
int **Membership_int;
};
struct archive_element *archive;
struct archive_element current;
struct archive_element new_pool1;
struct archive_element pool[softl];
struct archive_element new_pool;
int current_in_arc,pos,generations;
long seed;

double *euc_dist;
double *range_min, *range_max;
void main_process();
void mutation();
void WriteChromosome();
//void ComputeMembership_sym(struct archive_element *Chrom);
void ComputeMembership(struct archive_element *Chrom);
double ComputeFitnessFCM(struct archive_element *c);
double ComputeFitnessPBM(struct archive_element*);
double ComputeFitnessXB(struct archive_element*);
//double ComputeFitnessSym(struct archive_element*);
//double ComputeFitnessDB(struct archive_element*);
//double ComputeFitnessDunn(struct archive_element*);
//double ComputeFitnessGDunn(struct archive_element*);
//double ComputeFitnessMSR(struct archive_element *c);
//double ComputeFitnessRV(struct archive_element*);
//double ComputeFitnessZSCR(struct archive_element*);
//double ComputeFitnessCAnew(struct archive_element*);
//double ComputeFitnesssense(struct archive_element*);
//double ComputeFitnessspeci(struct archive_element*);
double find_dom(double *,double *);
void burn_in_period();
void clustering();
double FindDistance(double *, double *);
int flip(double);
int see_similar(struct archive_element);
int similarity_in_points();
void UpdateCenter(struct archive_element *);
void form_nondominated_archive();
void menu(void) ;
int compute_length();
double find_dom1(double *f1,double *f2);
void printarchive();
void process_current_dominates_new(void);
void process_new_dominates_current(void);
void process_new_current_nondominating(void);
void print_archive1();
void read_file();
void writechromosome(struct archive_element);
void InitializePop();
//void InitializePopulation();
void printclustering(FILE *c_ptr,struct archive_element *y);

main()
{

int i,m,j,k,count,dn,*flag2,ii;
int dim,p;
FILE *c_ptr, *plt;
char c_file_name[40];
double maxnearest_dist,mindist,dist,ob;
POP_SIZE=softl;
flag2=(int *)malloc(POP_SIZE*sizeof(int));
read_file(); // read contents of input file like a.txt and stores the 2D matrix in array.
dim=d;  //d: dimension of each data point to be clustered
/*euc_dist=(double *)malloc(n*sizeof(double));

  nPts = n;
  maxPts=n;
 queryPt = annAllocPt(dim);					// allocate query point
 dataPts = annAllocPts(maxPts, dim);			// allocate data points
  nnIdx = new ANNidx[knear];						// allocate near neigh indices
  dists = new ANNdist[knear];					// read data points
  printf("\n nPts=%d dim=%d\n\n",nPts,dim);

  for(i=0;i<nPts;i++)
	 {
	   for(j=0;j<dim;j++)
	     {
	      printf("\n %lf",points[i][j]);
	      dataPts[i][j]=points[i][j];
	     }

	}
  for(i=0;i<nPts;i++)
	  {
	     for(j=0;j<dim;j++)
	       printf("\n dataPts=%lf",dataPts[i][j]);
	  }

  kdTree = new ANNkd_tree(					// build search structure
					dataPts,		// the data points
					nPts,			// number of points
					dim);			// dimension of space



  maxnearest_dist=0;
  printf("\n\n here\n\n");
  for(i=0;i<n;i++)
     {

       mindist=999999;
       for(j=0;j<n;j++)
         {
	   if(i!=j)
	    {
	     dist=0;
	     for(p=0;p<d;p++)
	       dist=dist+pow((points[i][p]-points[j][p]),2);
	     dist=sqrt(dist);
	     if(mindist>dist)
	       mindist=dist;
	    }
	 }

      if(maxnearest_dist<mindist)
          maxnearest_dist=mindist;
    }
//  threshold=maxnearest_dist;
   threshold=1.2;
   printf("\n Max nearest distance=%lf\n\n",maxnearest_dist);  */
   printf("\n Enter maximum number of clusters ");
   scanf("%d",&MaxLen);
   printf("\n Enter minimum number of clusters ");
   scanf("%d",&MinLen);
   /*MaxLen=K;*/
   archive=(struct archive_element *)malloc(POP_SIZE*sizeof(struct
						archive_element));
for(i=0;i<POP_SIZE;i++)
{
if ( (archive[i].Membership = (double **) malloc( MaxLen* sizeof (double *))) == NULL)
	exit(0);
if ( (pool[i].Membership = (double **) malloc( MaxLen* sizeof (double *))) == NULL)
	exit(0);
for (j=0;j<MaxLen; j++){
	printf("\nEnd allocating Membership %d of size %d: ",i,n);
	if ( (archive[i].Membership[j] = (double *) malloc (n* sizeof(double))) == NULL)
		exit(0);
	if ( (pool[i].Membership[j] = (double *) malloc (n* sizeof(double))) == NULL)
		exit(0);
}
}

printf("\nEnter how many indices u want to optimize:"); // i.e. how many objective functions. 
scanf("%d",&total_index);
FitnessIndex=(int *)malloc(total_index*sizeof(int));
range_min=(double *)malloc(total_index*sizeof(double));
range_max=(double *)malloc(total_index*sizeof(double));

menu(); //menu of objective functions to choose
new_pool.Membership=(double **)malloc(MaxLen*sizeof(double *));
		for(i=0;i<MaxLen;i++)
		new_pool.Membership[i]=(double *)malloc(n*sizeof(double));
for(i=0;i<total_index;i++)
{
printf("\nEnter %dth index: ",i);
scanf("%d",&FitnessIndex[i]);
}
printf("\nstart");
printf("\nEnter seed : ");
scanf("%lf",&seed);
srand(seed);
InitializePop(); //Initially each cluster solution or archive_element is intialized by randomly chosen cluster centers by this function.
printf("\n After This Initialization phase");
printf("running 5");
//WriteChromosome();
for(m=0;m<POP_SIZE;m++)
{
	//printf("\n m=%d",m);
	for(i=0;i<10;i++)
	{              ComputeMembership(&(pool[m])); //for each initialized solution according to InitializePop() function membership matrix is created.
		       printf("Membership computed\n\n");
		       UpdateCenter(&(pool[m])); //cluster centers are updated
			//printf("\n i=%d\n");

	}
	//for(i=0;i<5;i++)
	//{
		//      ComputeMembership(&(pool[m]));
		 //     printf("Membership computed of sym\n\n");
		 //     UpdateCenter(&(pool[m]));
			//printf("\n i=%d\n");

	//}

}
printf("\n After Updation");


/*WriteChromosome();
getchar();
getchar();*/
form_nondominated_archive(); // for each solution its corresponding objective function values are calculated and set of initial non-dominated solutions are produced
POP_SIZE=hardl;
printf("\n After initialization phase");
printf("running 11");
printarchive();
printf("\n arcsize=%d",arcsize);
//getchar();

/*burn_in_period();*/
Temperature = TMax;
printf("\n tmax=%lf",TMax);
alpha=exp((log(TMin)-log(TMax))/MAX_GEN);
printf("\n alpha=%lf",alpha);
printf("running 12");
//getchar();
//getchar();
/*printf("\n %lf",Temperature);*/
printf("\n arcsize=%d",arcsize);
if(arcsize==1)
   r=0;
else
   r=rand()% arcsize;
printf("\n r=%d",r);
generations=1;
printf("running 13");
for(j=0;j<MaxLen;j++)
{
for(dn=0;dn<d;dn++)
	{
	current.z[j].x[dn]= archive[r].z[j].x[dn];
	//printf("\n %lf",current.z[j].x[dn]);
	}
}
//current.len=archive[r].len;
current.Membership=(double **)malloc(MaxLen*sizeof(double *));
for(i=0;i<MaxLen;i++)
current.Membership[i]=(double *)malloc(n*sizeof(double));
for(i=0;i<MaxLen;i++)
{
	for(j=0;j<n;j++)
	current.Membership[i][j]=archive[r].Membership[i][j];
}

Obj_current=(double *)malloc(total_index*sizeof(double));
Obj=(double *) malloc(total_index*sizeof(double));
for(k=0;k<total_index;k++)
	Obj_current[k]=area[r][k];
current_in_arc=1;pos=r;
while (Temperature > TMin) { 
printf("\n In the main loop");
for (generations = 0; generations < MAX_GEN; generations++)
{

		/*mutation_string();  /*  from pool to new_pool */
		mutation();
		for(i=0;i<5;i++)
		{
		        ComputeMembership(&new_pool);
			UpdateCenter(&new_pool);
			//printf("\n i=%d\n");

	         }
		/*ComputeMembership_sym(&new_pool);
		UpdateCenter(&new_pool);
		ComputeMembership_sym(&new_pool);
	        UpdateCenter(&new_pool);*/
		for(k=0;k<total_index;k++)
		{

		if (FitnessIndex[k] == 1) Obj[k] = ComputeFitnessXB(&new_pool);
		else if (FitnessIndex[k] == 2) Obj[k] = ComputeFitnessFCM(&new_pool);
		else if (FitnessIndex[k] == 3) Obj[k] = ComputeFitnessPBM(&new_pool);
		//else if (FitnessIndex[k] == 4) Obj[k] = ComputeFitnessSym(&new_pool);
              //  else if (FitnessIndex[k] == 5) Obj[k] = ComputeFitnessMSR(&new_pool);
		//else if (FitnessIndex[k] == 6) Obj[k] = ComputeFitnessRV(&new_pool);
              //  else if (FitnessIndex[k] == 7) Obj[k] = ComputeFitnessZSCR(&new_pool);
		//else if (FitnessIndex[k] == 8) Obj[k] = ComputeFitnessCAnew(&new_pool);
		//else if (FitnessIndex[k] == 9) Obj[k] = ComputeFitnesssense(&new_pool);
		//else if (FitnessIndex[k] == 10) Obj[k] = ComputeFitnessspeci(&new_pool);
		}

		/*for(k=0;k<total_index;k++)
		{
		printf("\n %lf",Obj[k]);
		} */
		//printf("The main procss has started!!! \n\n");
		main_process();
		//printf("\n Generation=%d",generations);
		//printf("The main procss has finished!!! \n\n");
	}
	Temperature=Temperature*alpha;
	printf("\n Temperature=%lf",Temperature);
}
printf("\n At end");


printf("\narcsize=%d\n",arcsize);
sprintf(c_file_name,"%s_plot_amosa_sym_vga_XBSym_mod.out",file);
plt=fopen(c_file_name,"w+");
for(ii=0;ii<arcsize;ii++)
{
                //ComputeMembership_sym(&archive[ii]);

		for(k=0;k<total_index;k++)
		{

		/*if (FitnessIndex[k] == 1) ob = ComputeFitnessXB(&archive[ii]);
		else if (FitnessIndex[k] == 2) ob = ComputeFitnessFCM(&archive[ii]);
		else if (FitnessIndex[k] == 3) ob = ComputeFitnessPBM(&archive[ii]);
		else if (FitnessIndex[k] == 4) ob =
		ComputeFitnessSym(&archive[ii]);*/
		fprintf(plt,"%lf ",area[ii][k]);
		}
fprintf(plt,"\n");

//printf("i=%d\n",i);
sprintf(c_file_name,"%s_clus%d_vga_sym_XBSym_mod",file,ii+1);

printf("\nWriting file clus%d\n",ii+1);
c_ptr = fopen(c_file_name,"w");
//ComputeMembership(x,old_pop_ptr->ind_ptr->length);
fprintf(c_ptr,"%d  %d  %d\n",n,d,archive[ii].len);
printclustering(c_ptr,&archive[ii]);
//fprintf(c_ptr,"\n");
fclose(c_ptr);
}
fclose(plt);
print_archive1();

}


void print_archive1()
{
int i,j,k,m,h,dn,ii,*computed_class,Index2,flag,minpoint,Di3,jj,cl,count;
double mk, *Di2,min,max,minkow;
int *flag1;
char file1[30];
strcpy(file1,file);
strcat(file1,"amosa_sym_vga_XBSym_mod.txt");
fpo=fopen(file1,"w+");
if(fpo==NULL)
{
printf("\n Error is opening output file");
exit(1);
}
printf("\n arcsize=%d",arcsize);
for(ii=0;ii<softl;ii++)
	{
		if ( (archive[ii].Membership_int = (int **) malloc( MaxLen* sizeof (int *))) == NULL)
			exit(0);
 		for (j=0;j< MaxLen;j++)
 			if ( (archive[ii].Membership_int[j] = (int *) malloc( n* sizeof (int))) == NULL)
				exit(0);
	}
for(i=0;i<arcsize;i++)
{
        ComputeMembership(&archive[i]);

	for (j=0;j<n;j++)
	{
		max = 0.0;
		for (k=0;k<MaxLen;k++)
		{
		 if(archive[i].z[k].x[0]!=DUMMY)
		 {
		 	archive[i].Membership_int[k][j]=0;
		 	if(archive[i].z[k].x[0]!=DUMMY)
		   	{
				if (archive[i].Membership[k][j]>max)
				{
				max = archive[i].Membership[k][j];
				cl=k;
				}
			}
		}
	    }

		archive[i].Membership_int[cl][j]=1;
	}
	//minkow=minkowsi_score1(archive[i].Membership_int);
        /*printf("\n Minkowski Score is=%lf",minkow);
        fprintf(fpo,"\n Minkowski Score=%lf",minkow);*/
	h=0;
	fprintf(fpo,"\n Number of clusters %d: ",archive[i].len);  //important
	count=0;
	for(k=0;k<MaxLen;k++)
	{
	  if(archive[i].z[k].x[0]!=DUMMY)
		   {

		count++;
		fprintf(fpo,"\n Points in cluster %d: ",count);
		for (dn=0;dn<d;dn++)
		    fprintf(fpo,"%lf \t",archive[i].z[k].x[dn]);
		 fprintf(fpo,"\n");
		for(j=0;j<n;j++)
		{

		if(archive[i].Membership_int[k][j]==1)
			{
			fprintf(fpo,"\n");
			for(dn=0;dn<d;dn++)
			{
			fprintf(fpo,"%lf",points[j][dn]);
			fprintf(fpo,"\t");
			}
			}
		}

		h++;
		}
	}
}
	fclose(fpo);

}
double find_dom1(double *f1,double *f2)
  {
   int i;
   double amo=1;
   for(i=0;i<total_index;i++)
    {
      if(fabs(f1[i]-f2[i])!=0)
        {
         amo=amo*(f1[i]-f2[i])/(range_max[i]-range_min[i]);
	 }
	/*printf("\n amo=%lf",amo);
	getchar();*/
    }
   return(amo);
 }
/*void burn_in_period()
 {
     int r;
     int i,h,count=0,dn,jj,k,j;
     double energy,pos_energy=0,ob1;
     double *func,*funcu;
     srand(seed);

     func=(double *)malloc(sizeof(double)*total_index);
     funcu=(double *)malloc(sizeof(double)*total_index);
     while(count<60)
       {
         r=(rand()%arcsize);
	 printf("\n r=%d",r);

	 for(h=0;h<total_index;h++)
          {
            funcu[h]=area[r][h];
          }
	 for(h=0;h<MaxLen;h++)
	    {
	      for(dn=0;dn<d;dn++)
	        current.z[h].x[dn]=archive[r].z[h].x[dn];
	    }
         mutation();
	 ComputeMembership_sym(&new_pool);
	 UpdateCenter(&new_pool);
	 ComputeMembership_sym(&new_pool);
	 UpdateCenter(&new_pool);
	 for(k=0;k<total_index;k++)
		{

		if (FitnessIndex[k] == 1) ob1 = ComputeFitnessXB(&new_pool);
		else if (FitnessIndex[k] == 2) ob1= ComputeFitnessFCM(&new_pool);
		else if (FitnessIndex[k] == 3) ob1= ComputeFitnessPBM(&new_pool);
		else if (FitnessIndex[k] == 4) ob1= ComputeFitnessSym(&new_pool);
                //else if (FitnessIndex[k] == 5) ob1= ComputeFitnessMSR(&new_pool);
		else if (FitnessIndex[j] == 6) ob1= ComputeFitnessRV(&new_pool);
                else if (FitnessIndex[j] == 7) ob1= ComputeFitnessZSCR(&new_pool);
		/*if(range_max[k]< ob1)
		   range_max[k]=ob1;
		 if(range_min[k]> ob1)
		   range_min[k]=ob1;*/
	/*	   func[k]=ob1;
		}

         energy=find_dom1(func,funcu);
	 if(energy>0)
	   {
	     pos_energy+=energy;
	     count++;

	   }

	}
	printf("\n pos_energy=%lf",pos_energy);
	pos_energy=(pos_energy/count);
	TMax=(pos_energy/log(2));
	printf("\n TMax=%lf",TMax);

}*/
/* The below function is main process */
void main_process()
{
  int count1=0,count2=0,k;
  for(k=0;k<total_index;k++)
    {
         if(Obj_current[k]<=Obj[k])
	         count1++;
         if(Obj_current[k]>=Obj[k])
	       count2++;
    }
  if(count1==total_index)
       {

	printf("\n Current dominates the newly generated one");
	process_current_dominates_new();
       }
  else if(count2==total_index)
     {
	printf("\n New solution dominates the current one");

	process_new_dominates_current();
     }
  else
     {
	printf("\n New and Current are non dominating to each other");

	process_new_current_nondominating();
     }
}


void mutation()
 {

  double f,del;
  double rnd,rand_lap;
  int i,j,k,dn,pick1,m,pick2,j2,MutationOccurred = 1,l, index,flag,count,min_clus,pick;

  int **points_in_clus,max_clus,clus1,clus2,index2,jj;
  int *total_pt_in_clus;
  double *intra_distance,max_intra,s2,minimum,dist,sum,*sum1;
  double mut_scal=0.2, b=12.5;
        new_pool.len=0;
	for(j=0;j<MaxLen;j++)
	{
	for(dn=0;dn<d;dn++) {
	        // printf("\t %lf",current.z[j].x[dn]);
		new_pool.z[j].x[dn]= current.z[j].x[dn];
		}
	 if(new_pool.z[j].x[0]!=DUMMY) new_pool.len++;
	}

	for(j=0;j<MaxLen;j++)
	{
	  if (new_pool.z[j].x[0] != DUMMY){   /*  valid position */
      			if(flip(PMUT)) {
     			       for(dn=0;dn<d;dn++) {
				   if (flip(MUTATE_NORMAL)){

	                                rnd=(rand()/(RAND_MAX+1.0));
	                                while(rnd==0)
	                                  rnd=(rand()/(RAND_MAX+1.0));
	                                rnd=rnd-0.5;
	                               //printf("\n rnd=%lf",rnd);

	                                if(rnd<0)
	                                    rand_lap=new_pool.z[j].x[dn]+b*log(1-2*fabs(rnd));
	                                else rand_lap=new_pool.z[j].x[dn]-b*log(1-2*fabs(rnd));
	                                new_pool.z[j].x[dn]=rand_lap;
					//printf("\n pool=%lf",new_pool.z[j].x[dn]);
	                                count=0;
	                                while((new_pool.z[j].x[dn]<min[dn] || new_pool.z[j].x[dn]>max[dn])&&(count<20))
	                                        {
						  //printf("\n here");
	                                          new_pool.z[j].x[dn] = current.z[j].x[dn];
	                                          rnd=(rand()/(RAND_MAX+1.0));
	                                           while(rnd==0)
	                                              rnd=(rand()/(RAND_MAX+1.0));
	                                                   rnd=rnd-0.5;
	                                              //printf("\n rnd=%lf",rnd);
	                                            //printf("\n here");
	                                            if(rnd<0)
	                                                  rand_lap=new_pool.z[j].x[dn]+b*log(1-2*fabs(rnd));
	                                            else rand_lap=new_pool.z[j].x[dn]-b*log(1-2*fabs(rnd));
	                                               //printf("\n rand_lap=%lf",rand_lap);
	                                            new_pool.z[j].x[dn]=rand_lap;
	                                            count++;
						    //getchar();
	                                     }
					if(count==20)
					 {
					   if(new_pool.z[j].x[dn]<min[dn])
					     new_pool.z[j].x[dn]= min[dn];
					   else if (new_pool.z[j].x[dn]>max[dn])
	                                     new_pool.z[j].x[dn]= max[dn];
					   }
				   }
			       }
			     }

			else{
			      if (flip(MUTATE_DELETE) && (new_pool.len>2)) {
     		    		for(dn=0;dn<d;dn++)
         				new_pool.z[j].x[dn]=DUMMY;
			      }
			}
     		}
		else{  /*mutating DUMMY position */
      		    if(flip(MUTATE_INSERT)) {
			pick = rand()%n; /*select an element */
     		    	for(dn=0;dn<d;dn++) {
         			/*pool[i].z[j].x[dn]=InputFile.elements[pick][dn] + del;*/
         			new_pool.z[j].x[dn]=points[pick][dn] ;
      				/*f=(frand() * 2) - 1.0;
      				del=new_pool.z[j].x[dn];
      				del = 0.2 * del * f;
         			new_pool.z[j].x[dn]=new_pool.z[j].x[dn]+del;*/
				//printf("\n new_pool.z[j].x[dn]=%lf",new_pool.z[j].x[dn]);
				//getchar();
      			}
		    }


		}
   	}
    }



 double find_dom(double *f1,double *f2)
  {
   int i;
   double amo=1;
   for(i=0;i<total_index;i++)
    {
      if(fabs(f1[i]-f2[i])!=0)
        {
	   //printf("\n f1[%d]=%lf f2[%d]=%lf",i,f1[i],i,f2[i]);
	   //printf("\n range_max[%d]=%lf range_min[%d]=%lf",i,range_max[i],i,range_min[i]);
           amo=amo*fabs(f1[i]-f2[i])/(range_max[i]-range_min[i]);/*amount of domination type 2*/
	   /*amount of domination type 3*/
	  //amo=amo*fabs((f1[i]-range_min[i])/(range_max[i]-range_min[i])-(f2[i]-range_min[i])/(range_max[i]-range_min[i]));

	 }
	//(range_max[i]-range_min[i])
	//printf("\n amo=%lf",amo);

    }
   //getchar();
   return(amo);
 }

void process_current_dominates_new(void)
{
 int dominated_by=0,i,count,j,k,u,v;
 double deldom=0.0,product=1,prob,ran,amount;
 deldom=0.0;
 for(i=0;i<arcsize;i++)
   {
     count=0;
     for(k=0;k<total_index;k++)
       {
        if(area[i][k]<=Obj[k])
		count++;
        }
     if(count==total_index)
	{
	 dominated_by++;
	 amount=find_dom(Obj,area[i]);

	 deldom=deldom+amount;
	}

   }

  if(current_in_arc==0)
   {
	amount=find_dom(Obj,Obj_current);
	deldom=deldom+amount;
	dominated_by++;
   }
   prob=1.0/(1.0+exp(deldom/(dominated_by*Temperature)));
   //printf("\n prob=%lf",prob);
   ran=frand();
   if(prob>=ran){
       //printf("\n New is selected");
       for(j=0;j<MaxLen;j++)
           {
             current.z[j]=new_pool.z[j];
            }
       // current.len=new_pool.len;
       for(u=0;u<MaxLen;u++)
	 {
	   for(v=0;v<n;v++)
	    current.Membership[u][v]=new_pool.Membership[u][v];
	}
      for(k=0;k<total_index;k++)
	{
	  Obj_current[k]=Obj[k];
	}
     current_in_arc=0;
   }

}

void process_new_dominates_current(void)
{
int dominated_by,dominates,i,j,k,h,count1,count2,loc,g,u,v;
double minimum,product,prob,ran,amount;
int min_point,dn;
struct archive_element * archive1;
double ** area1;
int *flag_dom=(int *)malloc(arcsize*sizeof(int));
if(current_in_arc==1) flag_dom[pos]=1;
minimum=9999999;
dominated_by=0;
dominates=0;
for(i=0;i<arcsize;i++)
{
	flag_dom[i]=0;
	count1=0;count2=0;
	for(k=0;k<total_index;k++)
	{
	if(area[i][k]<=Obj[k])
		count1++;
	if(area[i][k]>=Obj[k])
		count2++;
	}
	if(count1==total_index)
	{
	   dominated_by++;
	   product=find_dom(Obj,area[i]);

	   if(minimum>product)
		{
		minimum=product;
		min_point=i;
		}
	}
	if(count2==total_index)
	{
	   dominates++;
	   flag_dom[i]=1;
	}
}
if(current_in_arc==1) dominates--;

if(dominated_by==0 && dominates==0)
{

	if(current_in_arc==1)
	  loc=pos;
	else
	{
	  loc=arcsize;

	  /*if(arcsize>=POP_SIZE && arcsize<softl)
	   {
		archive=(struct archive_element *)realloc(archive,sizeof(struct
								archive_element));
		archive[arcsize].Membership=(double **)malloc(MaxLen*sizeof(double *));
		for(u=0;u<MaxLen;u++)
		{
		archive[arcsize].Membership[u]=(double *)malloc(n*sizeof(double));
		}
		area=(double **)realloc(area,sizeof(double));
		area[arcsize]=(double *)malloc(total_index*sizeof(double));
		arcsize++;
	    }
	   else */
	   if(arcsize>softl)
	     clustering();

	}

	for(j=0;j<MaxLen;j++)
	{
	 for(dn=0;dn<d;dn++)
	   {
	     archive[loc].z[j].x[dn]=new_pool.z[j].x[dn];
	     current.z[j].x[dn]=new_pool.z[j].x[dn];
	    }
	}
	//archive[loc].len=new_pool.len;
	//current.len=new_pool.len;
	for(u=0;u<MaxLen;u++)
	{
		for(v=0;v<n;v++)
		{
		archive[loc].Membership[u][v]=new_pool.Membership[u][v];
		current.Membership[u][v]=new_pool.Membership[u][v];
		}
	}
	for(k=0;k<total_index;k++)
	{
	  area[loc][k]=Obj[k];
	  if(range_max[k]< area[loc][k])
		   range_max[k]=area[loc][k];
	  if(range_min[k]> area[loc][k])
		   range_min[k]=area[loc][k];
	  Obj_current[k]=Obj[k];
	}
	current_in_arc=1;
	pos=loc;
	}
	else  if(dominated_by==0 && dominates>0)
	{
	archive1=(struct archive_element *)malloc(arcsize*sizeof(struct archive_element));
	area1=(double **)malloc(arcsize*sizeof(double *));
	for(i=0;i<arcsize;i++)
	  area1[i]=(double *)malloc(total_index*sizeof(double));
	for(h=0;h<arcsize;h++)
	{
		for(j=0;j<MaxLen;j++)
		{
		  for(dn=0;dn<d;dn++)
	           {
		      archive1[h].z[j].x[dn]=archive[h].z[j].x[dn];
		     }

		}
		//archive1[h].len=archive[h].len;
		archive1[h].Membership=(double **)malloc(MaxLen*sizeof(double *));
		for(i=0;i<MaxLen;i++)
		archive1[h].Membership[i]=(double *)malloc(n*sizeof(double));
		for(u=0;u<MaxLen;u++)
		{
		  for(v=0;v<n;v++)
			archive1[h].Membership[u][v]=archive[h].Membership[u][v];
		}
	        for(k=0;k<total_index;k++)
	      	 {
		   area1[h][k]=area[h][k];
		 }
	}

	g=0;

	for(h=0;h<arcsize;h++)
	{
	   if(flag_dom[h]==0)
	     {
		//archive[g].len= archive1[h].len;
		for(j=0;j<MaxLen;j++)
		{
		   for(dn=0;dn<d;dn++)
		     archive[g].z[j].x[dn]=archive1[h].z[j].x[dn];
		}

		for(u=0;u<MaxLen;u++)
		{
		   for(v=0;v<n;v++)
			archive[g].Membership[u][v]=archive1[h].Membership[u][v];
		}
		for(k=0;k<total_index;k++)
		{
		    area[g][k]=area1[h][k];
		}
		g++;
	  }
	}
	arcsize=g;
	for(i=0;i<arcsize;i++)
	  {
	    for(u=0;u<MaxLen;u++)
		{
		free(archive1[i].Membership[u]);
		}
	    free(archive1[i].Membership);

	    free(area1[i]);
	    }
	    free(archive1);
	    free(area1);
	/*if(arcsize>=POP_SIZE && arcsize<softl)
	{
	  archive=(struct archive_element *)realloc(archive,sizeof(struct
							archive_element));
	  archive[arcsize].Membership=(double **)malloc(MaxLen*sizeof(double *));
	  for(u=0;u<MaxLen;u++)
		archive[arcsize].Membership[u]=(double *)malloc(n*sizeof(double));
	  area=(double **)realloc(area,sizeof(double *));
	  area[arcsize]=(double *)malloc(total_index*sizeof(double));
	}
	else*/ if(arcsize>softl)   clustering();

	for(j=0;j<MaxLen;j++)
		{
		for(dn=0;dn<d;dn++)
	           {
		        current.z[j].x[dn]=new_pool.z[j].x[dn];
		        archive[arcsize].z[j].x[dn]=new_pool.z[j].x[dn];
		     }
		  }
	//archive[arcsize].len=new_pool.len;
	//current.len=new_pool.len;
	   for(u=0;u<MaxLen;u++)
		   {
		    for(v=0;v<n;v++)
		       {
		          archive[arcsize].Membership[u][v]=new_pool.Membership[u][v];
		          current.Membership[u][v]=new_pool.Membership[u][v];
		       }
		   }
	    for(k=0;k<total_index;k++)
		   {
		     area[arcsize][k]=Obj[k];
		     if(range_max[k]< area[arcsize][k])
		            range_max[k]=area[arcsize][k];
	              if(range_min[k]> area[arcsize][k])
		           range_min[k]=area[arcsize][k];
		     Obj_current[k]=Obj[k];
		    }
	   arcsize++;
	   pos=arcsize-1;
	   current_in_arc=1;
	}
    else if(dominated_by>0 && dominates==0)
	{
	   prob=1.0/(1.0+exp(-minimum));
	   ran=frand();
	   if(prob>=ran)
	    {
	      for(j=0;j<MaxLen;j++)
	        {
		  for(dn=0;dn<d;dn++)
	            {
	              current.z[j].x[dn]=archive[min_point].z[j].x[dn];
		     }
		   }
	    //current.len=archive[min_point].len;
	    for(u=0;u<MaxLen;u++)
	          {
	          for(v=0;v<n;v++)
		       current.Membership[u][v]=archive[min_point].Membership[u][v];
	          }
	    for(k=0;k<total_index;k++)
	       {
	         Obj_current[k]= area[min_point][k];
	       }
	     current_in_arc=1;
	     pos=min_point;
	    }
	  else
	   {
	     for(j=0;j<MaxLen;j++)
	       {
	        for(dn=0;dn<d;dn++)
	         {
		   current.z[j].x[dn]=new_pool.z[j].x[dn];
	       }
	      }
	     //current.len=new_pool.len;

	     for(u=0;u<MaxLen;u++)
		{
		for(v=0;v<n;v++)
		current.Membership[u][v]=new_pool.Membership[u][v];
		}
	     for(k=0;k<total_index;k++)
	        {
		  Obj_current[k]=Obj[k];
	         }
	     current_in_arc=0;
	}
    }

}


void process_new_current_nondominating(void)
{
int dominated_by=0,i,count1,count2,k,j,m,g,u,v;
int dominates=0,dn;
double deldom=0,product,prob,ran;
struct archive_element *archive1;
double **area1 ;
int *flag_dom=(int *)malloc(arcsize*sizeof(int));
for(i=0;i<arcsize;i++)
{
	count1=0;count2=0;
	flag_dom[i]=0;

	for(k=0;k<total_index;k++)
	{

	   if(area[i][k]<=Obj[k]) count1++;
	   if(area[i][k]>=Obj[k]) count2++;
	}
	if(count1==total_index)
	{
		dominated_by++;
		product=find_dom(Obj,area[i]);

	        deldom=deldom+product;

	}
	if (count2==total_index)
	{
		dominates++;
		flag_dom[i]=1;
	}
     }
printf("Check 1\n");
fflush(stdout);
	if(dominated_by>0 && dominates==0)
	{
	   prob=1.0/(1.0+exp(deldom/(dominated_by*Temperature)));
	   ran=frand();
	   if(prob>=ran)
	      {
	          for(j=0;j<MaxLen;j++)
		     {
		      for(dn=0;dn<d;dn++)
		          current.z[j].x[dn]=new_pool.z[j].x[dn];
		      }

	// current.len=new_pool.len;
	         for(u=0;u<MaxLen;u++)
		    {
		       for(v=0;v<n;v++)
		          current.Membership[u][v]=new_pool.Membership[u][v];
		     }

	         for(k=0;k<total_index;k++)
		   {
		        Obj_current[k]=Obj[k];
		    }
	         current_in_arc=0;
	   }
	}
	else if(dominated_by==0 && dominates==0)
	{

	   /*if(arcsize>=POP_SIZE && arcsize>softl)
	     {
	       archive=(struct archive_element *)realloc(archive,(1*sizeof(struct
								archive_element )));

	       archive[arcsize].Membership=(double **)malloc(MaxLen*sizeof(double *));
	       for(j=0;j<MaxLen;j++)
		  archive[arcsize].Membership[j]=(double *)malloc(n*sizeof(double));
	       area=(double **)realloc(area,sizeof(double *));
	       area[arcsize]=(double *)malloc(total_index*sizeof(double));
	     }
	     else*/ if(arcsize>softl)
	        clustering();
	printf("check 2\n");
fflush(stdout);
	for(j=0;j<MaxLen;j++)
	{
	  for(dn=0;dn<d;dn++)
	    {
	     archive[arcsize].z[j].x[dn]=new_pool.z[j].x[dn];
	     current.z[j].x[dn]=new_pool.z[j].x[dn];
	    }
	}
	// archive[arcsize].len=new_pool.len;

	// current.len=new_pool.len;
	for(u=0;u<MaxLen;u++)
		{
		for(v=0;v<n;v++)
		{
		current.Membership[u][v]=new_pool.Membership[u][v];
		archive[arcsize].Membership[u][v]=new_pool.Membership[u][v];
		}
		}
	for(k=0;k<total_index;k++)
	{
	  area[arcsize][k]=Obj[k];
	  if(range_max[k]< area[arcsize][k])
		            range_max[k]=area[arcsize][k];
	  if(range_min[k]> area[arcsize][k])
		           range_min[k]=area[arcsize][k];
	  Obj_current[k]=Obj[k];
	}
	arcsize++;
	current_in_arc=1;
	pos=arcsize-1;
printf("check 3\n");
fflush(stdout);
	}
   else if(dominated_by==0 && dominates>0)
	{
	  archive1=(struct archive_element *)malloc(arcsize*sizeof(struct
						archive_element));
	  area1=(double **)malloc(arcsize*sizeof(double));
	  for(i=0;i<arcsize;i++)
	    {

	     archive1[i].Membership=(double **)malloc(MaxLen*sizeof(double *));
	     for(j=0;j<MaxLen;j++)
		archive1[i].Membership[j]=(double *)malloc(n*sizeof(double));
	    }

	   for(k=0;k<arcsize;k++)
	     {
	       area1[k]=(double *)malloc(total_index*sizeof(double));
	     }
	   for(j=0;j<arcsize;j++)
	     {
	      for(m=0;m<MaxLen;m++)
	         {
	           for(dn=0;dn<d;dn++)
	                  {
	                    archive1[j].z[m].x[dn]=archive[j].z[m].x[dn];
	                  }
	           }
	       for(u=0;u<MaxLen;u++)
	          {
		        for(v=0;v<n;v++)
			        archive1[j].Membership[u][v]=archive[j].Membership[u][v];
		   }
	       //archive1[j].len=archive[j].len;
	       for(k=0;k<total_index;k++)
	          {
	            area1[j][k]=area[j][k];
	          }
printf("check 4\n");
fflush(stdout);
	    }
	g=0;
	for(i=0;i<arcsize;i++)
	{
	if(flag_dom[i]==0)
	{
	for(j=0;j<MaxLen;j++)
	  {
	   for(dn=0;dn<d;dn++)
	    {
	      archive[g].z[j].x[dn]=archive1[i].z[j].x[dn];
	     }
	   }
	for(u=0;u<MaxLen;u++)
		{
		for(v=0;v<n;v++)
			archive[g].Membership[u][v]=archive1[i].Membership[u][v];
		}
	// archive[g].len=archive1[i].len;
	for(k=0;k<total_index;k++)
	    area[g][k]=area1[i][k];
	g++;
printf("check 5\n");
fflush(stdout);
	}
	}
	for(i=0;i<arcsize;i++)
	  {
	    for(u=0;u<MaxLen;u++)
		{
		free(archive1[i].Membership[u]);
		}
	    free(archive1[i].Membership);

	    free(area1[i]);
	    }
	    free(archive1);
	    free(area1);

arcsize=g;
/*if(arcsize>=POP_SIZE && arcsize<softl)
	{
	archive=(struct archive_element *)realloc(archive,sizeof(struct archive_element));
	archive[arcsize].Membership=(double **)malloc(MaxLen*sizeof(double *));
	area=(double **)realloc(area,sizeof(double));
	area[arcsize]=(double *)malloc(total_index*sizeof(double));
	}
else*/ if(arcsize>softl)
	     clustering();
for(j=0;j<MaxLen;j++)
	{
	  for(dn=0;dn<d;dn++)
	    {
	       archive[arcsize].z[j].x[dn]=new_pool.z[j].x[dn];
	       current.z[j].x[dn]=new_pool.z[j].x[dn];
	      }
	}
printf("check 6\n");
fflush(stdout);
//archive[arcsize].len=new_pool.len;

//current.len=new_pool.len;
for(u=0;u<MaxLen;u++)
		{
		archive[arcsize].Membership[u]=(double *)malloc(n*sizeof(double));
		for(v=0;v<n;v++)
		{
		current.Membership[u][v]=new_pool.Membership[u][v];
		archive[arcsize].Membership[u][v]=new_pool.Membership[u][v];

		}
		}

printf("check 7\n");
fflush(stdout);
for(k=0;k<total_index;k++)
{
	Obj_current[k]=Obj[k];
	area[arcsize][k]=Obj[k];
	if(range_max[k]< area[arcsize][k])
		            range_max[k]=area[arcsize][k];
	              if(range_min[k]> area[arcsize][k])
		           range_min[k]=area[arcsize][k];
}
arcsize++;
current_in_arc=1;
pos=arcsize-1;
}

}

void menu(void)

{

printf("\n Enter 1 for ComputeFitnessXB(pool)");

printf("\n Enter 2 for ComputeFitnessFCM(pool)");

printf("\n Enter 3 for ComputeFitnessPBM(pool)");
//printf("\n Enter 4 for ComputeFitnessSym(pool)");
//printf("\n Enter 5 for ComputeFitnessMSR(pool)");
//printf("\n Enter 6 for ComputeFitnessRV(pool)");
//printf("\n Enter 7 for ComputeFitnessZSCR(pool)");
//printf("\n Enter 8 for ComputeFitnessCA(pool)");
//printf("\n Enter 9 for ComputeFitnesssense(pool)");
//printf("\n Enter 10 for ComputeFitnessspeci(pool)");
//    printf("\n Enter 4 for ComputeFitnessDB(pool)");
//
//    printf("\n Enter 5 for  ComputeFitnessDunn(pool)");
//
//    printf("\n Enter 6 for ComputeFitnessPE(pool)");
//
//    printf("\nEnter 7 for ComputeFitnessMPS(pool)");
//
//    printf("\nEnter 8 for ComputeFitnessSC(pool)");

}

void read_file()
{
int i,j,dump,nclass,jh;
char name[30],ch,name1[30];
FILE *ifp,*ifpp;

printf("\n Enter the input file name: ");
scanf("%s",name);
printf("\n Enter data file name");
scanf("%s",file);
if((ifp=fopen(name,"r"))==NULL)
{
printf("\n File not found");
exit(9);
}

/*printf("\n Enter the true class file name: ");
scanf("%s",name1);
if((ifpp=fopen(name1,"r"))==NULL)
{
printf("\n File not found");
exit(9);
}
trucls=(int *)malloc(n*sizeof (int *));
if(trucls==NULL)
{
printf("\n Error in memory allocation");
}
for(i=0;i<n;i++)
{
	fscanf(ifpp,"%lf",&trucls[i]);
}
printf("\n printing trucls file \n");
{
for(i=0;i<n;i++)
printf("%d\n",trucls[i]);
}


scanf("%d",&jh);*/





fscanf(ifp,"%d %d %d",&n,&d,&K);
printf("\n Number of points %d,features %d",n,d);
points=(double **)malloc(n*sizeof (double *));
fclass=(int *)malloc(n*sizeof (int));
for(i=0;i<n;i++)
points[i]=(double *)malloc(d*sizeof(double));


if(points==NULL)
{
printf("\n Error in memory allocation");
}

min=(double *)malloc(d*sizeof(double));

max=(double *)malloc(d*sizeof(double));

for(j=0;j<d;j++)

{
min[j]=99999.999;

max[j]=-99999.999;

}
for(i=0;i<n;i++)

{

	//fscanf(ifp,"%d",&dump);
	for(j=0;j<d;j++)

	{
	fscanf(ifp,"%lf",&points[i][j]);

	if(points[i][j]<min[j]) min[j]=points[i][j];

	if(points[i][j] >max[j]) max[j]=points[i][j];

	}
//	fscanf(ifp,"%d",&fclass[i]);
//	fclass[i]++;
       //fscanf(ifp,"%d",&fclass);
 	//fscanf(ifp,"%d",&fclass[i]);
      //while(fgetc(ifp) != '\n');

}

printf("\n Printing the min and max values: ");
for(j=0;j<d;j++)
{
	printf("axis%d min=%lf max=%lf\n",j,min[j],max[j]);
}
/*for(i=0;i<n;i++)
{
printf("\n");
for(j=0;j<d;j++)
	printf("%lf",points[i][j]);
//printf("\t %d",fclass[i]);
}
fclose(ifp);*/
//fclose(testp);

}

/*The bellow function is responsible for computing the length of the chormosome */
int compute_length(struct archive_element chrom)
{
int k, length=0;

for(k=0;k<MaxLen;k++)
if(chrom.z[k].x[0]!= DUMMY) length++;
return (length);
}

int flip(double prob)
{
double  i;
i=frand();
if((prob==1.0) || (i<prob))
return(1);
else return(0);
}

/*void UpdateCenter(struct archive_element *Chrom)
{
      int Index1, Index2, dn,i,count;
        double  *sum;
	sum=(double *)malloc(d*sizeof(double));
	for (Index1 = 0; Index1 < MaxLen; Index1++){

                       for(dn=0;dn<d;dn++)
		         {
			   sum[dn]=0;
			 }
                        count=0;
			for(Index2=0;Index2<n;Index2++){
			   if(Chrom->Membership[Index1][Index2]==1)
			    {
			     count++;
			     for(dn=0;dn<d;dn++)
			      {
			       sum[dn]=sum[dn]+points[Index2][dn];
		     		}
			   }
			 }
			for(dn=0;dn<d;dn++)
			 {
			   Chrom->z[Index1].x[dn] = sum[dn]/count;
			 }

		}

 }*/


void UpdateCenter(struct archive_element *Chrom)
{
int Index1, Index2, dn,i;
double SumMembership = 0.0, SumCenter[DIMENSION], Value, weight=2.0;


for (Index1 = 0; Index1 < MaxLen; Index1++){
  SumMembership = 0;
  if(Chrom->z[Index1].x[0]!=DUMMY)
   {

	for (dn = 0; dn < d; dn++)
		SumCenter[dn]=0;
	for (Index2 = 0; Index2 < n; Index2++){
		Value  = pow( Chrom->Membership[Index1][Index2],weight);
	SumMembership += Value;

	for (dn = 0; dn < d; dn++){
		SumCenter[dn] += Value * points[Index2][dn];
	}
   }
}
for (dn = 0; dn < d; dn++){
if (SumMembership>0)
   {
	Chrom->z[Index1].x[dn]=SumCenter[dn]/SumMembership;
	//printf("\n %lf",Chrom->z[Index1].x[dn]);
   }
//else printf("\n!!!Index1 = %d", Index1);
}
}
}



void WriteChromosome()

{
int l,m,dn;
for(m=0;m<POP_SIZE;m++)
{
printf("\n");
for(l=0;l<MaxLen;l++){

	printf("(");
	for(dn=0;dn<d;dn++)
		printf("%lf  ",pool[m].z[l].x[dn]);
	printf(") ");

}
}
}


void printarchive()
{
   int l,m,dn;
  printf("\n In the initialization phase");
  for(m=0;m<arcsize;m++)
   {

    for(dn=0;dn<total_index;dn++)
      printf("\narea[%d][%d]=%lf\n",m,dn,area[m][dn]);

   }
}


void ComputeMembership(struct archive_element *Chrom)
{
	int Index1, Index2, Index3, flag,minpoint,count;
	double Sum;
       double Di2[MaxLen], Di3,min;


	for (Index1 = 0; Index1 < n; Index1++){
		for (Index2 = 0; Index2 < MaxLen; Index2++){
		    if(Chrom->z[Index2].x[0]!=DUMMY)
		    {

                               Chrom->Membership[Index2][Index1]=0;
				Di2[Index2] =  FindDistance(Chrom->z[Index2].x,
                                                 points[Index1]);
                                //printf("\n Di2[%d]=%lf",Index2,Di2[Index2]);
			}
		  }

          	min=9999999;flag=0;
	 	for (Index2 = 0; Index2 < MaxLen && (!flag); Index2++){
		    if(Chrom->z[Index2].x[0]!=DUMMY)
		     {
                      if(!flag)
                      {
		       Chrom->Membership[Index2][Index1]=0;
		       if(Di2[Index2]==0)
			{
			  Chrom->Membership[Index2][Index1]=1;
                          flag=1;
			  minpoint=Index2;
                          break;

			}
		       else
			{
			  Di3=Di2[Index2];
			  if(min>Di3)
			   {
 		     	     min=Di3;
			     minpoint=Index2;
			   }
			 }
                       }
		 }
	     }

      Chrom->Membership[minpoint][Index1] = 1;

     }
    Chrom->len=0;
   for (Index2 = 0; Index2 < MaxLen;Index2++){
	if(Chrom->z[Index2].x[0]!=DUMMY)
		    {
		        Chrom->len++;
          		count=0;
         	 	for(Index1=0;Index1<n;Index1++){
              		if(Chrom->Membership[Index2][Index1]==1)
                  		count++;
            		}
	   		Chrom->index1[Index2]=count;
          		//printf("\n no of points in cluster %d is %d",Index2,count);
          }
	  }

    //printf("\n End Membership");
  }

/*void ComputeMembership(struct archive_element *Chrom)
{
int Index1, Index2, Index3, Flag = 1;
double Sum, Di1i2[MAX_LEN], Di1i3, weight = 2.0;

for (Index1 = 0; Index1 < n; Index1++){
for (Index2 = 0; Index2 < MaxLen; Index2++){
Di1i2[Index2] =  FindDistance(Chrom->z[Index2].x,points[Index1]);
}
for (Index2 = 0; Index2 < MaxLen; Index2++){
Flag = 1;
if (Di1i2[Index2]==0) /* point lies at the center*/
	/*Chrom->Membership[Index2][Index1] = 1.0;
else {
	Sum = 0;
	for (Index3 = 0; Index3 < MaxLen; Index3++){
	Di1i3=Di1i2[Index3];
	if (Di1i3 == 0){
	Flag = 0;
	break;
	}
	else
	Sum += pow( (Di1i2[Index2]/Di1i3), 2.0/(weight - 1));
	}
	if (!Flag)
	Chrom->Membership[Index2][Index1] = 0;
	else if ( Sum > 0)
	Chrom->Membership[Index2][Index1] = 1.0/Sum;
	else{

	getchar();
	}

}
}
}
}*/

/*void InitializePop()
{
  int i,j,m,l,dn,ran,choice,ran2;
  int *Element=(int *)malloc(n*sizeof(int));

  for(i=0;i<POP_SIZE;i++)
    {
     for(j=0;j<n;j++)
        Element[j]=0;
     //pool[i].len=MaxLen;
     for(l=0;l<MaxLen;l++) {

          do{
	     ran2=rand()%n;
            }while (Element[ran2] == 1);
         Element[ran2] = 1;
        //printf("\n ran2=%d",ran2);
        for(dn=0;dn<d;dn++){
	   pool[i].z[l].x[dn]=points[ran2][dn];
	  //printf("\n %lf",pool[i].z[l].x[dn]);
	}
     }
  }
  printf("\n After initialization \n\n");
}*/

void InitializePop()
{
 int i,j,m,l,dn,k,r,index,wrong,g_pos[MAX_LEN], Element[NO_OF_ELM],nb;
 double mind,delta;
 for(k=0;k<POP_SIZE;k++)
    {

/* printf("\nInitializing pool %d",k);*/
 	for(l=0;l<MaxLen;l++) {
  		pool[k].index1[l]=0;
  	for(dn=0;dn<d;dn++)
    		pool[k].z[l].x[dn]=DUMMY;
       }

 	if (MaxLen == MinLen) r = MaxLen;
 	else
   	r=rand()%(MaxLen-MinLen)+MinLen;

 	pool[k].len=r;
        printf("\n %d",pool[k].len);
 	r=rand()%MaxLen;g_pos[0]=r;
	printf("Running 1");
for(i=1;i<pool[k].len;i++)
  	{
   		do{ wrong=0;
       		    r=rand()%MaxLen;
       		    for(j=0;j<i;j++)
         		if(g_pos[j]==r) wrong=1;
     		}while(wrong==1);
   		g_pos[i]=r;
  	}
 printf("\n In initialization \n");
 //getchar();

 for(l=0;l<n;l++)
	Element[l] = 0;
printf("Running 2");

 for(l=0;l<pool[k].len;l++){
    do{
       r=rand()%n;
    }
    while (Element[r] == 1);
    Element[r] = 1;
    for(dn=0;dn<d;dn++){
	delta = frand(); delta = 0.0;
        pool[k].z[g_pos[l]].x[dn]=points[r][dn];

	/*if (rand()%2)
       		pool[k].z[g_pos[l]].x[dn]=InputFile.elements[r][dn]+delta;
	else
        	pool[k].z[g_pos[l]].x[dn]=InputFile.elements[r][dn]-delta; */
    }
 }
 printf("Running 3");

}
printf("Running 4");
}

int similarity_in_points()
{
int i,j,total_similar=0,count=0,dn;
for(i=0;i<n;i++)
{
for(j=(i+1);j<n;j++)
{
	count=0;
	for(dn=0;dn<d;dn++)
	{
		if(points[i][dn]==points[j][dn])
			count++;
	}
	if(count==d) total_similar++;
}
}
return(total_similar);
}

int see_similar(struct archive_element y)
{
int i,j,count=0,flag=0,dn;
for(i=0;i<MaxLen;i++)
{

	for(j=(i+1);j<MaxLen;j++)
		{

			count=0;
			for(dn=0;dn<d;dn++)
			{
			if(y.z[i].x[dn]==y.z[j].x[dn])
				count++;
			}
			if(count==d)
			{
			flag=1;
			return(flag);
			}
		}


	}
return(flag);
}

/* The following function will find the nondominated archive */
void form_nondominated_archive()
{
int i,j,count1=0,count2=0,k,g,u,v,dn,ii,hill_climbno=5,f,count,jj;
int *flag=(int *)malloc(POP_SIZE*sizeof(int));
double Obj1;
double **area1, *area2;
//double **xnew;
area1=(double **)malloc(POP_SIZE*sizeof(double));
for(i=0;i<total_index;i++)
        {

	  range_min[i]=999999.99;
	  range_max[i]=0;
	}
//fprintf(fpo,"\n\nIn form nondominated archive");
for(i=0;i<POP_SIZE;i++)
{
       /* for(j=0;j<MaxLen;j++)
	  {
	    if(pool[i].z[j].x[0]!=DUMMY)
	    {
	     for(dn=0;dn<d;dn++)
	       printf("\t%lf",pool[i].z[j].x[dn]);
	    }
	     printf("\n");
	    }

	  getchar();
	  getchar();    */

	area1[i]=(double *)malloc(total_index*sizeof(double));
	for(j=0;j<total_index;j++)
	{

	if (FitnessIndex[j] == 1) Obj1 = ComputeFitnessXB(&pool[i]);

	else if (FitnessIndex[j] == 2) Obj1 = ComputeFitnessFCM(&pool[i]);
	else if (FitnessIndex[j] == 3) Obj1 = ComputeFitnessPBM(&pool[i]);
	//else if (FitnessIndex[j] == 4) Obj1 = ComputeFitnessSym(&pool[i]);
	//else if (FitnessIndex[j] == 5) Obj1 = ComputeFitnessMSR(&pool[i]);
	//else if (FitnessIndex[j] == 6) Obj1 = ComputeFitnessRV(&pool[i]);
       // else if (FitnessIndex[j] == 7) Obj1 = ComputeFitnessZSCR(&pool[i]);
	//else if (FitnessIndex[j] == 8) Obj1 =ComputeFitnessCAnew(&pool[i]);
	//else if (FitnessIndex[j] == 9) Obj1 =ComputeFitnesssense(&pool[i]);
	//else if (FitnessIndex[j] == 10) Obj1 =ComputeFitnessspeci(&pool[i]);
	area1[i][j]=Obj1;
	// fprintf(fpo,"\n %lf",area1[i][j]);
	}

       flag[i]=0;
}
area2=(double *)malloc(total_index*sizeof(double));
 printf("\n IN Initialize ");
 //xnew=(double **) malloc(MaxLen*sizeof(double));
 /*for(i=0;i<MaxLen;i++)
   xnew[i]=(double *)malloc(d*sizeof(double)); */
     for(ii=0;ii<POP_SIZE;ii++)
      {
        for(jj=0;jj<hill_climbno;jj++)
          {

             for(i=0;i<MaxLen;i++)
	      {
              for(f=0;f<d;f++)
                {
                  current.z[i].x[f]=pool[ii].z[i].x[f];
		  //printf("\t %lf",current.z[i].x[f]);
                }
	      }
	      current.len=pool[ii].len;
	      mutation();
	     /*printf("\n After mutation \n");
	      for(j=0;j<MaxLen;j++)
	       {
	        for(dn=0;dn<d;dn++) {
		 printf("%lf\t",new_pool.z[j].x[dn]);
	           }
	        printf("\n");

	        }
	      getchar();
	      getchar();	*/
	      for(i=0;i<5;i++)
		{       ComputeMembership(&new_pool);
			UpdateCenter(&new_pool);

		}
             /* ComputeMembership_sym(&new_pool);
	      UpdateCenter(&new_pool);
	      ComputeMembership_sym(&new_pool);
	      UpdateCenter(&new_pool);*/
	     for(j=0;j<total_index;j++)
	       {

	        if (FitnessIndex[j] == 1) Obj1 = ComputeFitnessXB(&new_pool);

	        else if (FitnessIndex[j] == 2) Obj1 = ComputeFitnessFCM(&new_pool);
	        else if (FitnessIndex[j] == 3) Obj1 = ComputeFitnessPBM(&new_pool);
		//else if (FitnessIndex[j] == 4) Obj1 = ComputeFitnessSym(&new_pool);
              //  else if (FitnessIndex[j] == 5) Obj1 = ComputeFitnessMSR(&new_pool);
		//else if (FitnessIndex[j] == 6) Obj1 = ComputeFitnessRV(&new_pool);
               // else if (FitnessIndex[j] == 7) Obj1 = ComputeFitnessZSCR(&new_pool);
		//else if (FitnessIndex[j] == 8) Obj1 = ComputeFitnessCAnew(&new_pool);
		//else if (FitnessIndex[j] == 9) Obj1 = ComputeFitnesssense(&new_pool);
		//else if (FitnessIndex[j] == 10) Obj1 = ComputeFitnessspeci(&new_pool);
		/*if(range_max[j]< Obj1)
		   range_max[j]=Obj1;
		 if(range_min[j]> Obj1)
		   range_min[j]=Obj1;*/

	        area2[j]=Obj1;
	      // fprintf(fpo,"\n %lf",area1[i][j]);
	      }
              count=0;
              for(i=0;i<total_index;i++)
                {
                  if(area1[ii][i]>=area2[i])
                    count++;
                }
              if(count==total_index)
               {
                 for(i=0;i<MaxLen;i++)
	           {
                      for(f=0;f<d;f++)
                        {
                            pool[ii].z[i].x[f]=new_pool.z[i].x[f];
                        }
	            }

		  for(j=0;j<total_index;j++)
	           {
		     	area1[ii][j]=area2[j];
		   }
		  for(u=0;u<MaxLen;u++)
	           {
	            for(v=0;v<n;v++)
		         pool[ii].Membership[u][v]=new_pool.Membership[u][v];
	            }
               }

            }

	    //printf("\n ii=%d",ii);
       }
 printf("\n At the end of initialize solution\n\n");
for(i=0;i<POP_SIZE;i++)
{
	if(flag[i]==0)
		{
		for(j=i+1;j<POP_SIZE;j++)
			{
			if(flag[i]==0)
			{
			if(flag[j]==0)
			{
				count1=0;count2=0;
				for(k=0;k<total_index;k++)
				{
				if(area1[i][k]>=area1[j][k]) count1++;
				if(area1[i][k]<=area1[j][k]) count2++;
				}
				if(count1==total_index)
				flag[i]=1;
				else if(count2==total_index)
				flag[j]=1;

			}
			}
		}
	}
}

area=(double **)malloc(POP_SIZE*sizeof(double));

for(j=0;j<POP_SIZE;j++)

{
	area[j]=(double *)malloc(total_index*sizeof(double));

}
g=0;
// fprintf(fpo,"\n POP_SIZE=%d",POP_SIZE);

for(i=0;i<POP_SIZE;i++)

{
// fprintf(fpo,"\n flag[%d]=%d\n",i,flag[i]);

	if(flag[i]==0)
	{
	//archive[g].len=pool[i].len;
	// fprintf(fpo,"\n length=%d",archive[g].len);

	   for(k=0;k<MaxLen;k++)
	     {
	       for(dn=0;dn<d;dn++)
	           {
		    archive[g].z[k].x[dn]=pool[i].z[k].x[dn];
		     //printf("\n  %lf",archive[g].z[k].x[dn]);
		   }
	      }
	   for(u=0;u<MaxLen;u++)
	     {
	         for(v=0;v<n;v++)
		     archive[g].Membership[u][v]=pool[i].Membership[u][v] ;
	     }

	   for(j=0;j<total_index;j++)

	     {
		 area[g][j]=area1[i][j];
		 if(range_max[j]< area[g][j])
		   range_max[j]=area[g][j];
	          if(range_min[j]> area[g][j])
		   range_min[j]=area[g][j];
	        printf("\n area[%d][%d]=%lf",g,j,area[g][j]);
	     }

	 g++;
  }
}
printf("\n Size of g is:%d",g);
arcsize=g;

}


double ComputeFitnessMSR(struct archive_element *c)
{
int i,j,dn,s,p,q=0,w;
int cls[n][d],cls1[n][d];
double element = 0.0;
//double addition=0.0;
double mean_clus;
int array[size][size];
int trans_array[size][size];
int row,col;
double sumrow[size];
double sumcol[size];
double sum,variab;
double sum1;
double total;
double MSR,RV,num,den,ZSR,MSR1=0.0;
double AMSR[MaxLen];
int im,jm;
int column;
//new_start
int r,cl,count;
int k,l,a;
int arr[n][d];
int trans_arr[d][n];
double avg_row[size];
double avg_col[size];
double big_avg;
double big_sum;
int avgcl,avgr;
double tot;
double big_clus_sum = 0.0;
int d=9;
//new_stop
//print("Value of n and d id %f %f \n",n,d);
for(i=0;i<MaxLen;i++)
AMSR[i] = 0.0;

//printf("\ndddddddddddddddddddd=%d",d);
//scanf("%d",&a);

for(i=0;i<MaxLen;i++)
{

        big_avg=0.0;
       big_sum=0.0;
       avgcl=0.0,
	avgr=0.0;
       tot=0.0;


	if (c->z[i].x[0] != DUMMY)
			{
		for(dn=0;dn<d;dn++)
				{
					for(j=0;j<n;j++)
					cls[j][dn]=(int)(c->Membership[i][j]*points[j][dn]);
				}
		//printf("The %d th matrix before modify\n",i);
		for(j=0;j<n;j++)
		{
			for(dn=0;dn<d;dn++)
				{
				//	printf("row d= %d\n\n",d);
				}
			//printf("\n");
		}
//printf("\nabar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
count=0;
w=-1;
for(j=0;j<n;j++)
	{
	w++;
	count++;

	for(dn=0;dn<d;dn++)
		{
		if(cls[j][0]==0)
		{
			w--;
			count--;

			break;
		}
		else
		{
			cls1[w][dn]=cls[j][dn];
                        arr[w][dn] = cls1[w][dn];


		}
		}


	}

//printf("\nabar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
if(count!=0||w!=-1)
{
for(r=0;r<=w;r++)//average of row
{
avgr=0.0;
  for(cl=0;cl<d;cl++)
  {
   avgr = avgr + arr[r][cl];
  }
   avg_row[r] = (double)avgr/d;
//printf("\nabar abar abar dddddddddddddddddddd=%d",d);
}
//printf("\nabar abar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);


//column = 9;
//printf("Value of count: %d\n\n",count);
avgcl=0.0;
for(cl=0;cl<d;cl++)//average column
{

 for(r=0;r<count;r++)
 {
avgcl = avgcl + arr[r][cl];
 }
avg_col[cl]=(double)(avgcl/count);
//printf("\ncolumn:%d\n",d);
//printf("count=%d\n",count);
//printf("\navgcl = %lf",avgcl);
//printf("\navgcl = %lf and count=%d\n",avgcl,count);
avgcl=0.0;
}




//printf("\nabar abar abar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);

/*
for(cl=0;cl<d;cl++)//average column
{
 for(r=0;r<(w+1);r++)
 {
  avgcl = avgcl + trans_arr[cl][r];
 }
  avg_col[cl] = (double)(avgcl/(w+1));
}
*/

//start
/*for(r=0;r<=w;r++)
{
 for(cl=0;cl<d;cl++)
 {
printf("%d   ",arr[r][cl]);
}
printf("\n");
}
*/
//printf("\nTransposed arrar\n\n");
//transpose array


for(r=0;r<=w;r++)
{
 for(cl=0;cl<d;cl++)
 {
  trans_arr[cl][r]=arr[r][cl];
  big_sum = big_sum + arr[r][cl];
  //printf("%d   ",trans_arr[cl][r]);
 }//printf("\n");
}



for(cl=0;cl<d;cl++)
{
 for(r=0;r<=w;r++)
 {
  //trans_arr[cl][r]=arr[r][cl];
  //printf("%d   ",trans_arr[cl][r]);
 }//printf("\n");
}
//printf("\nValue of w:%d\n\n",(w+1));
big_avg = (double)big_sum/(d*count);
//scanf("%d",&l);

for(k=0;k<=w;k++)
{

 for(l=0;l<d;l++)
   {
     tot =(double)(tot+(pow((arr[k][l]-avg_col[l]-avg_row[k]+big_avg),2.0)));
   }

}
//printf("\ntotal has been calculated= %lf\n",tot);
AMSR[i] = tot/(count*d);

//printf("\n\nAMSR[%d] has been calculated=%lf\n\n",i,AMSR[i]);
 }
}
//scanf("%d",&a);
}

 if(i==MaxLen)
  {
    for(i=0;i<MaxLen;i++)
    {

    big_clus_sum =big_clus_sum + AMSR[i];
    }

  }
//printf("\n\nThe value of big_sum_clus= %f\n",big_clus_sum);
return (big_clus_sum);
}
//stop

double ComputeFitnessRV(struct archive_element *c)
{
int i,j,dn,s,p,q=0,w;
int cls[n][d],cls1[n][d];
double element = 0.0;
//double addition=0.0;
double mean_clus;
int array[size][size];
int trans_array[size][size];
int row,col;
double sumrow[size];
double sumcol[size];
double sum,variab;
double sum1;
double total;
double MSR,RV,num,den,ZSR,MSR1=0.0;
double ARV[MaxLen];
int im,jm;
int column;
//new_start
int r,cl,count;
int k,l,a;
int arr[n][d];
int trans_arr[d][n];
double avg_row[size];
double avg_col[size];
double big_avg;
double big_sum;
int avgcl,avgr;
double tot;
double big_clus_sum = 0.0;
int d=9;
//new_stop
//print("Value of n and d id %f %f \n",n,d);
for(i=0;i<MaxLen;i++)
ARV[i] = 0.0;

//printf("\ndddddddddddddddddddd=%d",d);
//scanf("%d",&a);

for(i=0;i<MaxLen;i++)
{

        big_avg=0.0;
       big_sum=0.0;
       avgcl=0.0,
	avgr=0.0;
       tot=0.0;


	if (c->z[i].x[0] != DUMMY)
			{
		for(dn=0;dn<d;dn++)
				{
					for(j=0;j<n;j++)
					cls[j][dn]=(int)(c->Membership[i][j]*points[j][dn]);
				}
		//printf("The %d th matrix before modify\n",i);
		for(j=0;j<n;j++)
		{
			for(dn=0;dn<d;dn++)
				{
				//	printf("row d= %d\n\n",d);
				}
			//printf("\n");
		}
//printf("\nabar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
count=0;
w=-1;
for(j=0;j<n;j++)
	{
	w++;
	count++;

	for(dn=0;dn<d;dn++)
		{
		if(cls[j][0]==0)
		{
			w--;
			count--;

			break;
		}
		else
		{
			cls1[w][dn]=cls[j][dn];
                        arr[w][dn] = cls1[w][dn];


		}
		}


	}

//printf("\nabar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
if(count!=0||w!=-1)
{
for(r=0;r<=w;r++)//average of row
{
avgr=0.0;
  for(cl=0;cl<d;cl++)
  {
   avgr = avgr + arr[r][cl];
  }
   avg_row[r] = (double)avgr/d;
//printf("\nabar abar abar dddddddddddddddddddd=%d",d);
}
for(k=0;k<=w;k++)
{

 for(l=0;l<d;l++)
   {
     tot =(double)(tot+(pow((arr[k][l]-avg_row[k]),2.0)));
   }

}

//printf("\ntotal has been calculated= %lf\n",tot);
ARV[i] = tot/(count*d);

//printf("\n\nAMSR[%d] has been calculated=%lf\n\n",i,AMSR[i]);
 }
}
//scanf("%d",&a);
}

 if(i==MaxLen)
  {
    for(i=0;i<MaxLen;i++)
    {

    big_clus_sum =big_clus_sum + ARV[i];
    }

  }
//printf("\n\nThe value of big_sum_clus= %f\n",big_clus_sum);
return (1/big_clus_sum);
}

double ComputeFitnessZSCR(struct archive_element *c)
{
int i,j,dn,s,p,q=0,w;
int cls[n][d],cls1[n][d];
double element = 0.0;
//double addition=0.0;
double mean_clus;
int array[size][size];
int trans_array[size][size];
int row,col;
double sumrow[size];
double sumcol[size];
double sum,variab;
double sum1;
double total;
double MSR,RV,num,den,ZSR,MSR1=0.0;
double AMSR[MaxLen],AZSR[MaxLen];
int im,jm;
int column;
//new_start
int r,cl,count;
int k,l,a;
int arr[n][d];
int trans_arr[d][n];
double avg_row[size];
double avg_col[size];
double big_avg;
double big_sum;
int avgcl,avgr;
double tot;
double big_clus_sum = 0.0;
int d=9;
//new_stop
//print("Value of n and d id %f %f \n",n,d);
for(i=0;i<MaxLen;i++)
AMSR[i] = 0.0;
AZSR[i] =0.0;

//printf("\ndddddddddddddddddddd=%d",d);
//scanf("%d",&a);

for(i=0;i<MaxLen;i++)
{

        big_avg=0.0;
       big_sum=0.0;
       avgcl=0.0,
	avgr=0.0;
       tot=0.0;


	if (c->z[i].x[0] != DUMMY)
			{
		for(dn=0;dn<d;dn++)
				{
					for(j=0;j<n;j++)
					cls[j][dn]=(int)(c->Membership[i][j]*points[j][dn]);
				}
		//printf("The %d th matrix before modify\n",i);
		for(j=0;j<n;j++)
		{
			for(dn=0;dn<d;dn++)
				{
				//	printf("row d= %d\n\n",d);
				}
			//printf("\n");
		}
//printf("\nabar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
count=0;
w=-1;
for(j=0;j<n;j++)
	{
	w++;
	count++;

	for(dn=0;dn<d;dn++)
		{
		if(cls[j][0]==0)
		{
			w--;
			count--;

			break;
		}
		else
		{
			cls1[w][dn]=cls[j][dn];
                        arr[w][dn] = cls1[w][dn];


		}
		}


	}

//printf("\nabar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);
if(count!=0||w!=-1)
{
for(r=0;r<=w;r++)//average of row
{
avgr=0.0;
  for(cl=0;cl<d;cl++)
  {
   avgr = avgr + arr[r][cl];
  }
   avg_row[r] = (double)avgr/d;
//printf("\nabar abar abar dddddddddddddddddddd=%d",d);
}
//printf("\nabar abar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);


//column = 9;
//printf("Value of count: %d\n\n",count);
avgcl=0.0;
for(cl=0;cl<d;cl++)//average column
{

 for(r=0;r<count;r++)
 {
avgcl = avgcl + arr[r][cl];
 }
avg_col[cl]=(double)(avgcl/count);
//printf("\ncolumn:%d\n",d);
//printf("count=%d\n",count);
//printf("\navgcl = %lf",avgcl);
//printf("\navgcl = %lf and count=%d\n",avgcl,count);
avgcl=0.0;
}




//printf("\nabar abar abar abar dddddddddddddddddddd=%d",d);
//scanf("%d",&a);

/*
for(cl=0;cl<d;cl++)//average column
{
 for(r=0;r<(w+1);r++)
 {
  avgcl = avgcl + trans_arr[cl][r];
 }
  avg_col[cl] = (double)(avgcl/(w+1));
}
*/

//start
/*for(r=0;r<=w;r++)
{
 for(cl=0;cl<d;cl++)
 {
printf("%d   ",arr[r][cl]);
}
printf("\n");
}
*/
//printf("\nTransposed arrar\n\n");
//transpose array


for(r=0;r<=w;r++)
{
 for(cl=0;cl<d;cl++)
 {
  trans_arr[cl][r]=arr[r][cl];
  big_sum = big_sum + arr[r][cl];
  //printf("%d   ",trans_arr[cl][r]);
 }//printf("\n");
}



for(cl=0;cl<d;cl++)
{
 for(r=0;r<=w;r++)
 {
  //trans_arr[cl][r]=arr[r][cl];
  //printf("%d   ",trans_arr[cl][r]);
 }//printf("\n");
}
//printf("\nValue of w:%d\n\n",(w+1));
big_avg = (double)big_sum/(d*count);
//scanf("%d",&l);

for(k=0;k<=w;k++)
{

 for(l=0;l<d;l++)
   {
     tot =(double)(tot+(pow((arr[k][l]-avg_col[l]-avg_row[k]+big_avg),2.0)));
   }

}
//printf("\ntotal has been calculated= %lf\n",tot);
AMSR[i] = tot/(count*d);
//if(count==1)
//scanf("%d",&a);
num=(((count-1)*(d-1))*(AMSR[i]-1));
den=sqrt(2*(count-1)*(d-1));
printf("\n\nvalue of num and den= %lf,%lf\n\n",num,den);
if(den==0.0||num==0.0)
{
AZSR[i]=0.0;
}
else
{
AZSR[i]=(double)(num/den);
}
//printf("\n\nAMSR[%d] has been calculated=%lf\n\n",i,AMSR[i]);
 }
}
//scanf("%d",&a);
}

 if(i==MaxLen)
  {
    for(i=0;i<MaxLen;i++)
    {

    big_clus_sum = big_clus_sum + AZSR[i];
    }

  }
if(big_clus_sum==0)
big_clus_sum=0.00001;
return (big_clus_sum);
}



double ComputeFitnessPBM(struct archive_element *c)
{
int i,j,k,dn,len,cl;
double ed,ed1,inter,max,min=9999999.999,fitns,product;
double s2,sum2,weight=2.0;
/*ComputeMembership_sym(c);
UpdateCenter(c);
ComputeMembership_sym(c);*/
//getchar();
/*UpdateCenter(c); */
ed=0.0;
len=0;

for(i=0;i<MaxLen;i++) {
	if (c->z[i].x[0] != DUMMY){ //ask
	len++;
	for(j=0;j<n;j++)
	{

			s2=0.0;
			for (dn=0;dn<d;dn++)
			{
				s2+=pow((c->z[i].x[dn]-points[j][dn]),2.0);//what points does: two dimensional

			}


			ed=ed + sqrt(s2)*pow(c->Membership[i][j],weight);

	}
    }
}
//printf("\n ed=%lf",ed);

inter=0.0;
for(i=0;i<MaxLen;i++) {
	 if (c->z[i].x[0] != DUMMY){
		for(k=0;k<MaxLen;k++) {
		 if (c->z[k].x[0] != DUMMY){
			if (i!= k){    /* valid center */
				sum2=0.0;
				for(dn=0;dn<d;dn++) {
					product=pow((c->z[k].x[dn]-c->z[i].x[dn]),2.0);
					sum2=sum2+product;
				}
				if(sum2>inter) inter=sum2;
			}
		}
	}
  }
  }

if(len>=2)
{
  fitns=(sqrt(inter))/(len*ed);
  }
  else fitns=0.000001;
  //printf("\n Fitness=%lf",fitns);
//printf("aaaa");
return(1.0/fitns);
}

double ComputeFitnessXB(struct archive_element *c)
{
int i,j,k,dn;
double s2,sum2,sum=0.0,min=9999999.99,dd,temp,nsum;
double XB=0.0;
c->len=0;
for(i=0;i<MaxLen;i++) {
	if (c->z[i].x[0] != DUMMY){
	c->len++;
		for(k=0;k<MaxLen;k++) {
	 		if (c->z[k].x[0] != DUMMY){
				if (i!= k){
				sum2=0.0;
					for (dn=0;dn<d;dn++) {
					dd=pow((c->z[k].x[dn]-c->z[i].x[dn]),2.0);
					sum2=sum2+dd;
							      }
		if(sum2<min) min=sum2;
    }
   }
}
}
}
//printf("\n len=%d",c->len);
//printf("\n minimum separation=%lf",sum2);
if (min == 0)/*two clusters centers same*/
{
 XB=9999999999.999;
 //printf("\n%lf\n",XB);
return(XB);
}
sum=0.0;nsum=0.0;
for(i=0;i<MaxLen;i++) {
	if (c->z[i].x[0] != DUMMY){
		for(j=0;j<n;j++){
		s2=0.0;
		for (dn=0;dn<d;dn++){
			s2+=pow((c->z[i].x[dn]-points[j][dn]),2.0);
   		}
		sum += c->Membership[i][j] * c->Membership[i][j] * s2;
	}
  }
  }
if(c->len>=2)
{
	XB = sum/(n * min);
	if (XB == 0) XB = 0.0000001;
	//printf("\nXB=%lf\n",XB);
}
else XB=99999.99;
return (XB);
}


double ComputeFitnessFCM(struct archive_element *c)
{
int i,j,k,dn;
double s2,sum=0.0;
double FCM=0.0;


sum=0.0;

for(i=0;i<MaxLen;i++) {
for(j=0;j<n;j++){
s2=0.0;

for (dn=0;dn<d;dn++){
	s2+=pow((c->z[i].x[dn]-points[j][dn]),2.0);
}
sum=sum+c->Membership[i][j] * c->Membership[i][j] * s2;
}
}
FCM = sum;
//printf("\nvalue of d in FCM=%d\n",d);
if (FCM == 0) FCM = 0.0000001;
return (FCM);

}

double ComputeFitnessCAnew(struct archive_element *c)
{
int i,j,k,fact1=0,fact2=0,tt=0;
double ca, cat;
for(i=0;i<n;i++)//for each point
{
	for(j=(i+1);j<n;j++)//for each next point
	{
		for(k=0;k<MaxLen;k++)//for each cluster
		{
		if(c->Membership[k][i]==1 && c->Membership[k][j]==1 && trucls[i]==trucls[j])// if both points resides in same cluster for both of true and clustering result solution
		{
			fact1++;//increase the first factor
			break;
		}
		else if(c->Membership[k][i]==1 && c->Membership[k][j]==0 || c->Membership[k][i]==0 && c->Membership[k][j]==1)//If both points resides in different clusters for our clustering solution
		{
			if(trucls[i]!=trucls[j])//in different clusters for true solution
			fact2++;//Increase second factor
			break;
		}
		}
	}
}


for(j=0;j<n;j++)
{
	for(i=(j+1);i<n;i++)
	{
	tt++; //total number of pairs
	}
}
ca=((fact1+fact2)/tt);
cat=(double)(1/ca);
return(cat);
}
double ComputeFitnesssense(struct archive_element *c)
{
int i,j,k,fact1=0,fact2=0;
double sense, sensec;
for(i=0;i<n;i++)//for each point
{
	for(j=(i+1);j<n;j++)//for each next point
	{
		for(k=0;k<MaxLen;k++)//for each cluster
		{
		if(c->Membership[k][i]==1 && c->Membership[k][j]==1 && trucls[i]==trucls[j])// if both points resides in same cluster for both of true and clustering result solution
		{
			fact1++;//increase the first factor
			break;
		}
		else if(c->Membership[k][i]==1 && c->Membership[k][j]==0 || c->Membership[k][i]==0 && c->Membership[k][j]==1)//If both points resides in different clusters for our clustering solution
		{
			if(trucls[i]==trucls[j])//in different clusters for true solution
			fact2++;//Increase second factor
			break;
		}
		}
	}
}
sense=(double)(fact1/(fact1+fact2));
sensec=(double)(1/sense);
return(sensec);
}

double ComputeFitnessspeci(struct archive_element *c)
{
int i,j,k,fact1=0,fact2=0;
double speci,specif;
for(i=0;i<n;i++)//for each point
{
	for(j=(i+1);j<n;j++)//for each next point
	{
		for(k=0;k<MaxLen;k++)//for each cluster
		{
		if(c->Membership[k][i]==1 && c->Membership[k][j]==1 && trucls[i]!=trucls[j])// if both points resides in same cluster for both of true and clustering result solution
		{
			fact1++;//increase the first factor
			break;
		}
		else if(c->Membership[k][i]==1 && c->Membership[k][j]==0 || c->Membership[k][i]==0 && c->Membership[k][j]==1)//If both points resides in different clusters for our clustering solution
		{
			if(trucls[i]!=trucls[j])//in different clusters for true solution
			fact2++;//Increase second factor
			break;
		}
		}
	}
}
speci=(double)(fact2/(fact2+fact1));
specif=(double)(1/speci);
return(specif);
}

double FindDistance(double *x, double *y)
{
double distance=0,sum=0;
int i;
for(i=0;i<d;i++)
{
	sum=sum+pow((x[i]-y[i]),2);
}
/*printf("\n%lf",sum);*/
distance=sqrt(sum);
return(distance);
}

void writechromosome(struct archive_element y)
{
int i,j;
printf("\n\n");
for(i=0;i<MaxLen;i++)
{
printf("(");
for(j=0;j<d;j++)
	{
	printf("%lf",y.z[i].x[j]);
	}
printf(")");
}
}

void printclustering(FILE *c_ptr,struct archive_element *y)//important
{
    int i,j,l,count=0,counts,count1,count2,clus=0,k,dn,bj=0,is,p,knear=2,tu;
    double sum=0.0,s2,s3,maxx,DB,intra,mindist,dist1,temp,silho;
    double *ds;
  int cl;
	double db,min;
  float max;
double dis[MaxLen][n];
double sum1,sum2;
double ddis[n][MaxLen],final[MaxLen][n];
double totalsum=0.0;

  for (j=0;j<n;j++)
    {
      max = 0.0;
      for (i=0;i<MaxLen;i++)
	{
	  if (y->Membership[i][j]>max){
	    max = y->Membership[i][j];
	    cl=i;
	  }
	}
      for (i=0;i<d;i++)
	fprintf(c_ptr,"%f   ",points[j][i]);
        fprintf(c_ptr,"%d\n",cl+1);

    }
//Start computing DB index of current cluster
ds=(double *)malloc(MaxLen*sizeof(double));
for(j=0;j<MaxLen;j++)
ds[j]=0.0;
    for(j=0;j<MaxLen;j++) {//loop1

	if (y->z[j].x[0] != DUMMY){//newloop
	    count=0;
          for(is=0;is<n;is++)
	      {//loop2
	       s2=0.0;
                if(y->Membership[j][is]==1)
		 {//loop3
           		count++;
			    for (dn=0;dn<d;dn++){
			      s2=s2+pow((points[is][dn]-y->z[j].x[dn]),2);
						 }
                        sum=sum+sqrt(s2);

		}//end loop3


               }// end loop2

              if(count!=0)
                {
                 ds[j]=(sum/count);
                 }
              else ds[j]=0;
             // printf("\nds[%d]=%lf",j,ds[j]);
         }//end newloop
        }//loop1
counts=0;
sum=0.0;
      for(i=0;i<MaxLen;i++) {   //lo1
	if (y->z[i].x[0] != DUMMY){//newloop1
             counts++;
             maxx=0;
             for(j=0;j<MaxLen;j++)
                {//lo2
		if (y->z[j].x[0] != DUMMY){//newloop2
                 if(j!=i)
                   {//lo3
                  s2=0.0;
                       for (dn=0;dn<d;dn++){//lo4
			    s2=s2+pow((y->z[i].x[dn]-y->z[j].x[dn]),2);
			 } //end lo4
                       s2=sqrt(s2);
                       s3=(ds[i]+ds[j])/s2;
                       if (s3>maxx)  maxx=s3;

               	   }// end lo3
			}//end of newloop2
                }// end lo2
             sum=sum+maxx;
			}//end of newloop1

         }//end lo1
        //printf("\n sum=%lf",sum);
        DB=sum/counts;

fprintf(c_ptr,"DB= \n%lf\n",DB);
//Start computing Silhautte index
//1st compute intra cluster average separation for each element for its corresponding cluster

for(i=0;i<MaxLen;i++)
{
	if(y->z[i].x[0]!=DUMMY)
	{
		for(j=0;j<n;j++)
		{
	         	if(y->Membership[i][j]==1)
			{
			count=0;
			sum1=0.0;
				for(k=0;k<n;k++)
				{
					if(y->Membership[i][k]==1 && j!=k)
					{
					count++;
					temp=FindDistance(points[j],points[k]);
					sum1=sum1+temp;
					}
				}
				if(count==0)
				dis[i][j]=0.0;
				else
				dis[i][j]=double(sum1/count);
			}
		}
	}
}
//now compute minimum inter cluster average distance for each element of each cluster

for(i=0;i<MaxLen;i++)
{
	if(y->z[i].x[0]!=DUMMY)
	{

		for(k=0;k<n;k++)
		{
			if(y->Membership[i][k]==1)
			{
			min=99999999.99;

				for(j=0;j<MaxLen;j++)
				{
					if(y->z[j].x[0]!=DUMMY && i!=j)
					{
					count1=0;
					sum2=0.0;
						for(l=0;l<n;l++)
						{
							if(y->Membership[j][l]==1)
							{
							count1++;
							temp=FindDistance(points[k],points[l]);
							sum2=sum2+temp;
							}
						}
					ddis[k][j]=double(sum2/count1);
					if(ddis[k][j]<min)
					min=ddis[k][j];
					}
				}
			final[i][k]=min;
			}
		}
	}
}

//now compute actual silhoutte index

for(i=0;i<MaxLen;i++)
{
	if(y->z[i].x[0]!=DUMMY)
	{
	clus++;
	count2=0;
	sum=0.0;
		for(j=0;j<n;j++)
		{
			if(y->Membership[i][j]==1)
			{
			count2++;
			if(dis[i][j]<final[i][j])
			temp=final[i][j];
			else
			temp=dis[i][j];
			sum+=((final[i][j]-dis[i][j])/temp);
			//printf("dis[%d][%d]=%lf, final[%d][%d]=%lf\n",i,j,i,j,dis[i][j],final[i][j]);
			}
		}
//	printf("sum=%lf, count2=%d\n",sum,count2);
	totalsum+=double(sum/count2);
	}
}
silho=double(totalsum/clus);
//printf("totalsum=%lf\n",totalsum);
//printf("Clus=%d",clus);
fprintf(c_ptr,"Silho= \n%lf\n",silho);

}

void clustering()
     {
        int i,j,k,l,h,p,g2,g,pp,dn,uu,vv,jj,hh;
        double **area1;
	struct archive_element * archive1;
	int *point1, *point2;
	int u=0,v=0,w,m;
	double *dist,min, **distance;
        int no_clus=arcsize;
	int *flag;
        int *cluster=(int *)malloc(sizeof(int)*(softl+1));
        double **arc2;

        archive1=(struct archive_element *)malloc
				((softl+2)*sizeof(struct archive_element));
	area1=(double **)malloc((softl+2)*sizeof(double *));
	for(i=0;i<(softl+2);i++)
	  area1[i]=(double *)malloc(total_index*sizeof(double));
	for(h=0;h<arcsize;h++)
	{
		archive1[h].Membership=(double **)malloc(MaxLen*sizeof(double *));
		for(i=0;i<MaxLen;i++)
		    archive1[h].Membership[i]=(double *)malloc(n*sizeof(double));

	 }
	printf("check 9\n");
fflush(stdout);
        point1=(int *)malloc(sizeof(int)*(softl+1));
        point2=(int *)malloc(sizeof(int)*(softl+1));
        dist=(double *)malloc(sizeof(double)*softl);

        distance=(double **)malloc((softl+1)*sizeof(double *));
	k=arcsize;
        printf("\n IN CLUSTERING");
        printf("\n arcsize=%d",arcsize);
        for(i=0;i<(softl+1);i++)
           distance[i]=(double *)malloc((softl+1)*sizeof(double));
        for(i=0;i<arcsize;i++)
              cluster[i]=i;
        for(i=0;i<k;i++)
           {
              distance[i][i]=2000000;
              for(j=i+1;j<k;j++)
                 {
		   distance[i][j]=0.0;
                   for(p=0;p<total_index;p++)
		    {
                      distance[i][j]=distance[i][j]+pow((area[i][p]-area[j][p]),2);
		     }
                   distance[j][i]=sqrt(distance[i][j]);
                  }

             }
  printf("check 10\n");
fflush(stdout);

         flag=(int *)malloc((softl+1)*sizeof(int));
         while(no_clus>hardl)
            {
                min=2000000;
                for(i=0;i<arcsize;i++)
                   {
                     flag[i]=0;
                    }
                for(i=0;i<k;i++)
                   {
                     for(j=0;j<k;j++)
                       {
                        if(j!=k)
                        {
                         if(min>distance[i][j])
                            {
                              min=distance[i][j];
                              u=i;
                              v=j;
                             }
                        }
                     }
                   }
                if(cluster[u]==u && cluster[v]==v)
                   {
                     cluster[u]=v;
                     cluster[v]=u;
                    }
                 else if(cluster[u]==u)
                    {
                      j=cluster[v];
                      while(cluster[j]!=v)
                          {
                            j=cluster[j];
                           }
                       cluster[j]=u;
                       cluster[u]=v;
                     }
                  else if(cluster[v]==v)
                     {
                        j=cluster[u];
                        while(cluster[j]!=u)
                          {
                            j=cluster[j];
                          }
                         cluster[j]=v;
                         cluster[v]=u;
                       }
                   else
                      {
                         j=cluster[u];
                         while(cluster[j]!=u)
                            {
                              j=cluster[j];
                             }
                          cluster[j]=v;
                          p=cluster[v];
                          while(cluster[p]!=v)
                             {
                               p=cluster[p];
                              }
                           cluster[p]=u;
                      }
                    no_clus=no_clus-1;
                    g=0;
                    point1[g]=u;
                    j=cluster[u];
                    while(j!=u)
                        {
                          g++;
                          point1[g]=j;
                          j=cluster[j];
                         }
                     for(i=0;i<=g;i++)
                         {
                            w=point1[i];
                            flag[w]=1;
                            for(j=i+1;j<=g;j++)
                              {

                                m=point1[j];
                                flag[m]=1;
                                distance[m][w]= distance[w][m]=2000000;

                               }
                          }
                       for(i=0;i<arcsize;i++)
                           {
                             if(flag[i]==0)
                              {  /*see its's end bracket*/
                               if(cluster[i]==i)
                                  {
                                     w=point1[0];
                                     min=distance[w][i];
                                     for(j=1;j<=g;j++)
                                        {
					   m=point1[j];
                                           if(min>distance[m][i])
                                                min=distance[m][i];
                                           }
                                       for(j=0;j<=g;j++)
                                          {
                                            m=point1[j];
                                            distance[m][i]=min;
                                           }
                                        flag[i]=1;
                                     }
                                  else
                                     {
                                         g2=0;
                                         point2[g2]=i;j=cluster[i];
                                         while(j!=i)
                                             {
                                               g2++;
                                               point2[g2]=j;
                                               j=cluster[j];
                                              }
                                          w=point1[0];
                                          m=point2[0];
                                          min=distance[w][m];
                                          for(j=0;j<=g;j++)
                                            {
                                              w=point1[j];
                                              for(p=0;p<=g2;p++)
                                               {
                                                m=point2[p];
                                                if(min>distance[w][m])
                                                    min=distance[w][m];
                                                }
                                              }
                                          for(j=0;j<=g;j++)
                                               {
                                                 for(p=0;p<=g2;p++)
                                                      {
                                                        w=point1[j];
                                                        m=point2[p];
                                                        distance[m][w]=distance[w][m]=min;
                                                        flag[m]=1;
                                                       }
                                                    }

                                          }
                                      }
                                 }
                             }
			 for(hh=0;hh<arcsize;hh++)
	                    {

		                for(jj=0;jj<MaxLen;jj++)
		                 {
		                    for(dn=0;dn<d;dn++)
		                          archive1[hh].z[jj].x[dn]=archive[hh].z[jj].x[dn];
		                  }

		                 for(uu=0;uu<MaxLen;uu++)
		                   {
		                     for(vv=0;vv<n;vv++)
			                  archive1[hh].Membership[uu][vv]=archive[hh].Membership[uu][vv];
		                   }
		                  for(uu=0;uu<total_index;uu++)
		                     {
		                      area1[hh][uu]=area[hh][uu];
		                    }

	                       }
	printf("check 11\n");
fflush(stdout);

                  for(i=0;i<arcsize;i++)
                    {
                     flag[i]=0;
                     }
                  k=0;
                  for(i=0;i<arcsize;i++)
                    {
		     if(flag[i]==0)
		     {
                     if(cluster[i]!=i)
                       {
                        g=0;
                        point1[g]=i;
                        flag[i]=1;
                        j=cluster[i];
                        while(j!=i)
                          {
                           g++;
                           point1[g]=j;
                           flag[j]=1;
                           j=cluster[j];

                          }

                        for(j=0;j<=g;j++)
                          {
                           dist[j]=0;
                           w=point1[j];
                           for(p=0;p<=g;p++)
                              {
			       if(p!=j)
			       {
                                  m=point1[p];
			         for(pp=0;pp<total_index;pp++)
                                   dist[j]=dist[j]+pow((area[w][pp]-area[m][pp]),2);
				 dist[j]=sqrt(dist[j]);
                              }
			     }
                           }

                        min=dist[0];
                        w=point1[0];
                        for(j=1;j<=g;j++)
                          {
                           if(min>dist[j])
                            {
                             min=dist[j];
                             w=point1[j];
                            }
                          }

		for(jj=0;jj<MaxLen;jj++)
		{
		   for(dn=0;dn<d;dn++)
		     archive[k].z[jj].x[dn]=archive1[w].z[jj].x[dn];
		}

		for(uu=0;uu<MaxLen;uu++)
		{
		   for(vv=0;vv<n;vv++)
			archive[k].Membership[uu][vv]=archive1[w].Membership[uu][vv];
		}
		for(uu=0;uu<total_index;uu++)
		{
		    area[k][uu]=area1[w][uu];
		}
		k++;

            }
           else
               {
	            for(jj=0;jj<MaxLen;jj++)
		       {
		           for(dn=0;dn<d;dn++)
		          archive[k].z[jj].x[dn]=archive1[i].z[jj].x[dn];
		      }

		     for(uu=0;uu<MaxLen;uu++)
		       {
		           for(vv=0;vv<n;vv++)
			      archive[k].Membership[uu][vv]=archive1[i].Membership[uu][vv];
		       }
		     for(uu=0;uu<total_index;uu++)
		        {
		           area[k][uu]=area1[i][uu];
		        }

                     k++;
                  }
                }
	      }
              arcsize=k;
              printf("\n arcsize=%d",arcsize);
	      printf("\n afterclustering:");
	      /*for(i=0;i<arcsize;i++)
              {
                printf("\n %f",area[i][0]);
                printf("\t %f",area[i][1]);
              }*/
          }
