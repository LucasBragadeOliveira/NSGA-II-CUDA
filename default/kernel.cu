#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define tam_pop = 32

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("\nError at %s:%d | %s\n",__FILE__,__LINE__,cudaGetErrorString(cudaGetLastError()));\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("\nError at %s:%d | %s\n",__FILE__,__LINE__,cudaGetErrorString(cudaGetLastError()));\
    return EXIT_FAILURE;}} while(0)

__global__ void setup_rand(curandState *state, unsigned long seed)
{
	int tid = threadIdx.x;
	curand_init(seed+tid, tid, 0, &state[tid]);
}

__global__ void gen_pop0(curandState *globalState, float *populacao)
{
	int tid = threadIdx.x;
	curandState localState = globalState[tid];
	float random = curand_uniform(&localState);
	populacao[tid] = 70 + random * 70;
	globalState[tid] = localState;
}

__global__ void individuo_teste(float *populacao){

	populacao[0] = 75.0864;
	populacao[1] = 138.9080;
	populacao[2] = 107.5178;
	populacao[3] = 86.9607;
	populacao[4] = 90.4996;
	populacao[5] = 72.8030;
}

__global__ void custo(float *populacao, float* custo, int offset){
	int tid = threadIdx.x + offset;

	float individuo[6];
	float vazao = populacao[(tid * 6)] + populacao[(tid * 6) + 1] + populacao[(tid * 6) + 2] + populacao[(tid * 6) + 3] + populacao[(tid * 6) + 4] + populacao[(tid * 6) + 5];

	individuo[0] = populacao[(tid * 6)];
	individuo[1] = populacao[(tid * 6) + 1];
	individuo[2] = populacao[(tid * 6) + 2];
	individuo[3] = populacao[(tid * 6) + 3];
	individuo[4] = populacao[(tid * 6) + 4];
	individuo[5] = populacao[(tid * 6) + 5];

	//CALCULO DE PERDA HIDRAULICA
	//CONSTANTES
	float pi = 3.14;
	float g = 9.8;
	float l1 = 160;
	float l2a = 91.6;
	float l2b = 86.26;
	float l2c = 82.54;
	float l2d = 80.58;
	float l3 = 13.4;
	float d1 = 6.6;
	float d2 = 6.2;
	float r = 70000;
	float k = 0.2;
	
	//CALCULANDO RUGOSIDADE TOTAL
	float e1 = k / d1;
	float e2 = k / d2;

	//CALCULANDO A AREA (m^2)
	float a1 = pi*(powf(d1,4) / 4);
	float a2 = pi*(powf(d2,4) / 4);

	//CALCULANDO FATOR DE ATRITO F (MAIOR SECAO)
	float a = powf((64 / r), 8);
	float b = e1 / (3.7*d1);
	float c = (5.74 / powf(r, 0.9))*-1;
	float d = 2500 / (-1.0*r);
	float potd = powf(d, 6);
	float aux = b + c;
	float caln = logf(aux);
	float parteb = 9.5*(powf((caln - potd), -16));
	float fatorf1 = powf((a + parteb), 0.125);

	//CALCULANDO FATOR DE ATRITO F (MENOR SECAO)
	float a22 = powf((64 / r), 8);
	float b22 = e2 / (3.7*d2);
	float c2 = (5.74 / powf(r, 0.9))*-1;
	float d22 = 2500 / (-1.0*r);
	float potd2 = powf(d22, 6);
	float aux2 = b22 + c2;
	float caln2 = logf(aux2);
	float parteb2 = 9.5*(powf((caln2 - potd2), -16));
	float fatorf2 = powf((a22 + parteb2), 0.125);

	//CALCULANDO FATOR DE ATRITO F TOTAL
	float ftotal = fatorf1 + fatorf2;

	//FOR
	float vm[6];

	float p1a[6];
	float p1b[6];
	float p2a[6];
	float p2b[6];
	float p3a[6];
	float p3b[6];
	float p4a[6];
	float p4b[6];
	float p5a[6];
	float p5b[6];
	float p6a[6];
	float p6b[6];
	float p7a[6];
	float p7b[6];

	float pt1[6];
	float pt2[6];
	float pt3[6];
	float pt4[6];
	float pt5[6];
	float pt6[6];
	float pt7[6];

	float pcond18[6];
	float pcond27[6];
	float pcond36[6];
	float pcond45[6];

	float ca1 = 0.080 + (0.080 * 0.03); //Fator K para curva de 28º
	float ca2 = 0.100 + (0.100 * 0.03); //Fator K para curva de 30º

	float cb1 = 0.030 + (0.030 * 0.03); //Fator K para curva de 22º
	float cb2 = 0.020 + (0.020 * 0.03); //Fator K para curva de 21

	float cc1 = 0.051 + (0.051 * 0.03); //Fator K para curva de 16º
	float cc2 = 0.047 + (0.047 * 0.02); //Fator K para curva de 12º

	float cd1 = 0.0120 + (0.012 * 0.02); //Fator K para curva de 4º
	float cd2 = 0.0118 + (0.0118 * 0.02); //Fator K para curva de 3º

	float pc1[6];
	float pc2[6];
	float pc3[6];
	float pc4[6];

	float perda1[6];
	float perda2[6];
	float perda3[6];
	float perda4[6];

	float potencia[6];
	float hl;
	float potencia_total = 0;

	int i;

	for (i = 0; i < 6; i++){

		//VM 
		vm[i] = ((individuo[i] / a1) + (individuo[i] / a2)) / 2;

		//CALCULANDO PERDAS DE CARGA TUBO	
		p1a[i] = ftotal*((l1 / d1)*((powf(vm[i], 2) / 2) * g));
		p1b[i] = ftotal*((l1 / d2)*((powf(vm[i], 2) / 2) * g));

		p2a[i] = ftotal*((l1 / d1)*((powf(vm[i], 2) / 2) * g));
		p2b[i] = ftotal*((l1 / d2)*((powf(vm[i], 2) / 2) * g));

		p3a[i] = ftotal*((l2a / d1)*((powf(vm[i], 2) / 2) * g));
		p3b[i] = ftotal*((l2a / d2)*((powf(vm[i], 2) / 2) * g));

		p4a[i] = ftotal*((l2b / d1)*((powf(vm[i], 2) / 2) * g));
		p4b[i] = ftotal*((l2b / d2)*((powf(vm[i], 2) / 2) * g));

		p5a[i] = ftotal*((l2c / d1)*((powf(vm[i], 2) / 2) * g));
		p5b[i] = ftotal*((l2c / d2)*((powf(vm[i], 2) / 2) * g));

		p6a[i] = ftotal*((l2d / d1)*((powf(vm[i], 2) / 2) * g));
		p6b[i] = ftotal*((l2d / d2)*((powf(vm[i], 2) / 2) * g));

		p7a[i] = ftotal*((l3 / d1)*((powf(vm[i], 2) / 2) * g));
		p7b[i] = ftotal*((l3 / d2)*((powf(vm[i], 2) / 2) * g));

		pt1[i] = p1a[i] + p1b[i];
		pt2[i] = p2a[i] + p2b[i];
		pt3[i] = p3a[i] + p3b[i];
		pt4[i] = p4a[i] + p4b[i];
		pt5[i] = p5a[i] + p5b[i];
		pt6[i] = p6a[i] + p6b[i];
		pt7[i] = p7a[i] + p7b[i];

		pcond18[i] = pt1[i] + pt2[i] + pt6[i] + pt7[i];
		pcond27[i] = pt1[i] + pt3[i] + pt6[i] + pt7[i];
		pcond36[i] = pt1[i] + pt4[i] + pt6[i] + pt7[i];
		pcond45[i] = pt1[i] + pt5[i] + pt6[i] + pt7[i];

		//CALCULANDO PERDAS NAS CURVAS POR CONDUTOS

		pc1[i] = (ca1 * (powf(vm[i], 2) / (2 * g))) + (ca2 * (powf(vm[i], 2) / (2 * g)));
		pc2[i] = (cb1 * (powf(vm[i], 2) / (2 * g))) + (cb2 * (powf(vm[i], 2) / (2 * g)));
		pc3[i] = (cc1 * (powf(vm[i], 2) / (2 * g))) + (cc2 * (powf(vm[i], 2) / (2 * g)));
		pc4[i] = (cd1 * (powf(vm[i], 2) / (2 * g))) + (cd2 * (powf(vm[i], 2) / (2 * g)));

		perda1[i] = pcond18[i] + pc1[i];
		perda2[i] = pcond27[i] + pc2[i];
		perda3[i] = pcond36[i] + pc3[i];
		perda4[i] = pcond45[i] + pc4[i];

		if (i == 0){
			//saida_perdas[i] = perda1[i];
			hl = 54 - perda1[i];
		}
		else if (i == 1){
			//saida_perdas[i] = perda2[i];
			hl = 54 - perda2[i];
		}
		else if (i == 2){
			//saida_perdas[i] = perda3[i];
			hl = 54 - perda3[i];
		}
		else if (i == 3){
			//saida_perdas[i] = perda4[i];
			hl = 54 - perda4[i];
		}
		else if (i == 4){
			//saida_perdas[i] = perda4[i];
			hl = 54 - perda4[i];
		}
		else if (i == 5){
			//saida_perdas[i] = perda3[i];
			hl = 54 - perda3[i];
		}

		potencia[i] = 0.0098*(0.1463 + 0.018076*hl + 0.0050502*individuo[i] - 0.000035254*hl*individuo[i] - 0.00012337*powf(hl, 2) - 0.000014507*powf(individuo[i], 2))*hl*individuo[i];
		potencia_total += potencia[i];
	}
	
	//penalidades
	float lambda = 5;
	float F;

	F = lambda * fmaxf(0, (powf((320 - potencia_total), 2)));

	float f1 = -1 * potencia_total + F;
	f1 = f1 / vazao;

	c = 109.175;

	float f2 = sqrtf(powf((individuo[0] - c), 2) + powf((individuo[1] - c), 2) + powf((individuo[2] - c), 2) + powf((individuo[3] - c), 2) + powf((individuo[4] - c), 2) + powf((individuo[5] - c), 2));

	custo[tid * 2] = f1;
	custo[(tid * 2) + 1] = f2;

	printf("\nINDVIDUO %d = %f  %f  %f  %f  %f  %f \n CUSTO = %f %f \n\n",tid , populacao[(tid * 6)], populacao[(tid * 6) + 1], populacao[(tid * 6) + 2], populacao[(tid * 6) + 3], populacao[(tid * 6) + 4], populacao[(tid * 6) + 5], f1, f2);
}

__global__ void init_fronts(unsigned int *tam_fronts, unsigned int *fronts){
	int tid = threadIdx.x;

	fronts[tid] = 999999;
	tam_fronts[tid] = 0;
}

__global__ void NDS1(float *populacao, float *custo,unsigned int *ctd_domi, bool *conj_domi,unsigned int *rank, int n_pop,
					 unsigned int *tam_fronts, unsigned int *fronts, unsigned int *index_front){
	int tid = threadIdx.x;
	int n_domi_local = 0;

	bool conj_local[10];

	float custo1 = custo[(tid*2)];
	float custo2 = custo[(tid*2)+1];

	__shared__ unsigned int n_1afront;
	__shared__ unsigned int index_1afront;
	n_1afront = 0;
	index_1afront = 0;

	for (int i = 0; i < n_pop; i++){
		conj_local[i] = false;
	}
	
	for (int i = 0; i < n_pop; i++){

		/*
		if (i == tid){
			continue;
		}
		*/

		float q_custo1 = custo[i*2];
		float q_custo2 = custo[(i*2)+1];

		if (((custo1 <= q_custo1) && (custo2 <= q_custo2)) && ((custo1 < q_custo1) || (custo2 < q_custo2))){
			conj_local[i] = true;
		}
		else if (((q_custo1 <= custo1) && (q_custo2 <= custo2)) && ((q_custo1 < custo1) || (q_custo2 < custo2))){
			n_domi_local++;
		}
	}

	__syncthreads();

	int pos_1afront;

	if (n_domi_local == 0){
		rank[tid] = 1;
		atomicAdd(&n_1afront, 1);
		n_domi_local = 999999;
		__syncthreads();
		pos_1afront = atomicAdd(&index_1afront, 1);
		fronts[pos_1afront] = tid;
		index_front[0] = index_1afront;
	}

	tam_fronts[0] = n_1afront;
	printf("\nFronteira 1 index %d = %d %d %d %d %d %d %d %d %d %d", index_1afront, fronts[0], fronts[1], fronts[2], fronts[3], fronts[4], fronts[5], fronts[6], fronts[7], fronts[8], fronts[9]);

	for (int i = 0; i < n_pop; i++){
		conj_domi[(tid * n_pop) + i] = conj_local[i];
	}
	ctd_domi[tid] = n_domi_local;

	printf("\n(%d) | Custo %f %f | N Domina = %d |Conjunto %d%d%d%d%d%d%d%d%d%d", tid, custo1, custo2, n_domi_local, conj_local[0], conj_local[1], conj_local[2], conj_local[3], conj_local[4], conj_local[5], conj_local[6], conj_local[7], conj_local[8], conj_local[9]);
	//printf("\n\n(%d) | Custo %f %f | N Domina = %d |Conjunto %d%d%d%d%d%d%d%d%d%d", tid, custo1, custo2, ctd_domi[tid], conj_domi[(tid * 6) + 0], conj_domi[(tid * 6) + 1], conj_domi[(tid * 6) + 2], conj_domi[(tid * 6) + 3], conj_domi[(tid * 6) + 4], conj_domi[(tid * 6) + 5], conj_domi[(tid * 6) + 6], conj_domi[(tid * 6) + 7], conj_domi[(tid * 6) + 8], conj_domi[(tid * 6) + 9]);
}

__global__ void NDS2_neto(unsigned int n_pop, unsigned int individuo, unsigned int *ctd_domi, bool *conj_domi){
	unsigned int tid = threadIdx.x;

	atomicSub(&ctd_domi[tid], (unsigned int)conj_domi[(individuo * n_pop) + tid]);
}

__global__ void NDS2_filho(unsigned int rank, unsigned int n_pop,float *populacao, bool *conj_domi, unsigned int *ctd_domi,
						   unsigned int *pop_rank, unsigned int *tam_fronts, unsigned int *fronts, unsigned int *index_front){
	unsigned int tid = threadIdx.x;
	unsigned int individuo;
	unsigned int i;
	unsigned int index_inicial = 0;
		
	if (rank != 2){
		int i = 0;
		do{		
			index_inicial += tam_fronts[i];
			i++;
		} while (i <= rank - 3);
	}

	individuo = fronts[tid + index_inicial];
	//printf("\n n_pop %d", n_pop);
	printf("\n rank %d index inicial %d", rank, index_inicial);
	printf("\n Antes Filho %d Individuo %d | %d %d %d %d %d %d %d %d %d %d", tid, individuo, ctd_domi[0], ctd_domi[1], ctd_domi[2], ctd_domi[3], ctd_domi[4], ctd_domi[5], ctd_domi[6], ctd_domi[7], ctd_domi[8], ctd_domi[9]);
	NDS2_neto<<<1,n_pop>>>(n_pop, individuo, ctd_domi, conj_domi);
	cudaDeviceSynchronize();
	/*
	for (i = 0; i < n_pop; i++){
		//ctd_domi[i] = ctd_domi[i] - (unsigned int)conj_domi[(individuo * n_pop) + i];		
		atomicSub(&ctd_domi[i], (unsigned int)conj_domi[(individuo * n_pop) + i]);
		if (ctd_domi[i] == 0){
		atomicAdd(&tam_fronts[rank - 1], 1);
		int pos = atomicAdd(index_front, 1);
		fronts[pos] = individuo;
		pop_rank[i] = rank;
		}
	}
	*/
	printf("\n Depois Filho %d Individuo %d | %d %d %d %d %d %d %d %d %d %d", tid, individuo, ctd_domi[0], ctd_domi[1], ctd_domi[2], ctd_domi[3], ctd_domi[4], ctd_domi[5], ctd_domi[6], ctd_domi[7], ctd_domi[8], ctd_domi[9]);
}

__global__ void NDS2_insert(unsigned int rank,unsigned int *ctd_domi,unsigned int *pop_rank,unsigned int *tam_fronts,
							unsigned int *fronts,unsigned int *index_front){
	unsigned int tid = threadIdx.x;
	unsigned int pos;
	if (ctd_domi[tid] == 0){
		printf("\n Insert individuo %d | rank %d", tid,rank);
		ctd_domi[tid] = 999999;
		atomicAdd(&tam_fronts[rank - 1], 1);
		pos = atomicAdd(&index_front[0], 1);
		fronts[pos] = tid;
		pop_rank[tid] = rank;
		printf("\n Insert individuo %d | pop_rank %d | pos %d | index %d", tid, pop_rank[tid], pos, index_front[0]);
	}
}

__global__ void NDS2_pai(int n_pop, float *populacao, bool *conj_domi, unsigned int *ctd_domi, unsigned int *pop_rank,
						 unsigned int *tam_fronts, unsigned int *fronts, unsigned int *index_front){
	unsigned int rank = 2;
	unsigned int pop_com_front = tam_fronts[0];
	unsigned int index_tam_fronts = 0;
	unsigned int i;

	while (pop_com_front < n_pop){
		printf("\n\nFronteira %d = %d %d %d %d %d %d %d %d %d %d", (rank - 1), fronts[0], fronts[1], fronts[2], fronts[3], fronts[4], fronts[5], fronts[6], fronts[7], fronts[8], fronts[9]);
		printf("\nTam Fronts %d %d %d %d %d %d %d %d %d %d", tam_fronts[0], tam_fronts[1], tam_fronts[2], tam_fronts[3], tam_fronts[4], tam_fronts[5], tam_fronts[6], tam_fronts[7], tam_fronts[8], tam_fronts[9]);
		NDS2_filho<<<1, tam_fronts[index_tam_fronts]>>>(rank, n_pop, populacao, conj_domi, ctd_domi, pop_rank, tam_fronts, fronts, index_front);
		cudaDeviceSynchronize();
		printf("\nCtd domi %d %d %d %d %d %d %d %d %d %d", ctd_domi[0], ctd_domi[1], ctd_domi[2], ctd_domi[3], ctd_domi[4], ctd_domi[5], ctd_domi[6], ctd_domi[7], ctd_domi[8], ctd_domi[9]);
		NDS2_insert<<<1, n_pop>>>(rank, ctd_domi,pop_rank,tam_fronts,fronts,index_front);
		cudaDeviceSynchronize();
		printf("\nRanks %d %d %d %d %d %d %d %d %d %d", pop_rank[0], pop_rank[1], pop_rank[2], pop_rank[3], pop_rank[4], pop_rank[5], pop_rank[6], pop_rank[7], pop_rank[8], pop_rank[9]);
		/*
		for (i = 0; i < n_pop; i++){
			if (ctd_domi[i] == 0){
				ctd_domi[i] = 999999;
				tam_fronts[rank - 1]++;
				fronts[index_front[0]] = i;
				pop_rank[i] = rank;
				index_front[0]++;
			}
		}
		*/
		rank++;
		index_tam_fronts++;
		pop_com_front = pop_com_front + tam_fronts[index_tam_fronts];
	}
	printf("\n\nFronteira %d = %d %d %d %d %d %d %d %d %d %d", (rank - 1), fronts[0], fronts[1], fronts[2], fronts[3], fronts[4], fronts[5], fronts[6], fronts[7], fronts[8], fronts[9]);
	printf("\nTam Fronts %d %d %d %d %d %d %d %d %d %d", tam_fronts[0], tam_fronts[1], tam_fronts[2], tam_fronts[3], tam_fronts[4], tam_fronts[5], tam_fronts[6], tam_fronts[7], tam_fronts[8], tam_fronts[9]);
	printf("\nRanks %d %d %d %d %d %d %d %d %d %d", pop_rank[0], pop_rank[1], pop_rank[2], pop_rank[3], pop_rank[4], pop_rank[5], pop_rank[6], pop_rank[7], pop_rank[8], pop_rank[9]);
}

__global__ void bitonic_sort(float *valores, unsigned int *chaves, unsigned int tam,bool flag_impar){
	//printf("\nbitonic inicio");
		extern __shared__ float shared[];
		float* sh_valor = (float*)&shared;
		float* sh_chave = (float*)&shared[tam];

		unsigned int tid = threadIdx.x;
		float tmp;

		if (flag_impar && tid+1 == tam){
			sh_valor[tid] = 999999;
			sh_chave[tid] = -1;
		}
		else{
			sh_valor[tid] = valores[tid];
			sh_chave[tid] = chaves[tid];
			__syncthreads();
		}

		for (int k = 2; k <= tam; k *= 2){
			for (int j = k / 2; j>0; j /= 2){
				int ixj = tid ^ j;
				if (ixj > tid){
					if ((tid & k) == 0){
						if (sh_valor[tid] > sh_valor[ixj]){
							tmp = sh_valor[tid];
							sh_valor[tid] = sh_valor[ixj];
							sh_valor[ixj] = tmp;

							tmp = sh_chave[tid];
							sh_chave[tid] = sh_chave[ixj];
							sh_chave[ixj] = tmp;
						}
					}
					else{
						if (sh_valor[tid] < sh_valor[ixj]){
							tmp = sh_valor[tid];
							sh_valor[tid] = sh_valor[ixj];
							sh_valor[ixj] = tmp;

							tmp = sh_chave[tid];
							sh_chave[tid] = sh_chave[ixj];
							sh_chave[ixj] = tmp;
						}
					}
				}
				__syncthreads();
			}
		}
		if (flag_impar && tid+1 == tam){
			shared[tid] = 999999;
		}
		else{
			valores[tid] = sh_valor[tid];
			chaves[tid] = sh_chave[tid];
		}
		//printf("\nbitonic fim");
}

__global__ void dist_multidao(int n_pop, unsigned int *pop_rank, float *pop_custo, float *pop_dist, unsigned int *tam_fronts,
							  unsigned int *fronts, float *sort_custo1, float *sort_custo2, unsigned int *sort_chaves1,
							  unsigned int *sort_chaves2){

	unsigned int tid = threadIdx.x;
	printf("multidao inicio");
	__shared__ unsigned int sh_fronts[10];
	__shared__ unsigned int sh_tam_fronts[10];
	__shared__ float sh_pop_dist[10];

	unsigned int index_inicio = 0;	
	unsigned int index_fim;
	unsigned int individuo;
	unsigned int local_rank;
	unsigned int pos_c1;
	unsigned int pos_c2;

	sh_fronts[tid] = fronts[tid];
	sh_tam_fronts[tid] = tam_fronts[tid];
	individuo = sh_fronts[tid];

	sort_chaves1[tid] = sh_fronts[tid];
	sort_chaves2[tid] = sh_fronts[tid];
	sort_custo1[tid] = pop_custo[individuo * 2];
	sort_custo2[tid] = pop_custo[(individuo * 2) + 1];

	local_rank = pop_rank[individuo];

	int i = 0;

	while (i+1 != local_rank){
		index_inicio += sh_tam_fronts[i];
		i++;
	}

	index_fim = index_inicio + sh_tam_fronts[local_rank-1] - 1;
	
	float old_custo1[10];
	float old_custo2[10];
	float old_chave1[10];
	float old_chave2[10];

	if (tid == 0){
		for (i = 0; i < n_pop; i++){
			old_custo1[i] = sort_custo1[i];
			old_custo2[i] = sort_custo2[i];
			//printf("o1 %d o2 %d sh %d\n", old_chave1[i], old_chave2[i], sh_fronts[i]);
		}
	}

	if (individuo == sh_fronts[index_inicio] && sh_tam_fronts[local_rank-1] > 1){
		//printf("\nbitonic call");
		bool impar = (bool)(sh_tam_fronts[local_rank-1]%2);

		int tam_sort = sh_tam_fronts[local_rank-1];
		if (impar){
			tam_sort++;
		}
		bitonic_sort<<<1, tam_sort, (sizeof(float)*tam_sort*2)>>>(&sort_custo1[index_inicio], &sort_chaves1[index_inicio], tam_sort, impar);
		bitonic_sort<<<1, tam_sort, (sizeof(float)*tam_sort*2)>>>(&sort_custo2[index_inicio], &sort_chaves2[index_inicio], tam_sort, impar);
		
	}
	cudaDeviceSynchronize();
	
	printf("\nTid %d, Ind %d, rank %d, inicio %d, fim %d", tid, individuo, pop_rank[individuo], index_inicio, index_fim);
	
	if (tid == 0){
		printf("\n Custo 1 \n");
		for (i = 0; i < n_pop; i++){
			printf("%d|%f  ", sort_chaves1[i], old_custo1[i]);
		}
		printf("\n");
		for (i = 0; i < n_pop; i++){
			printf("%d|%f  ", sort_chaves1[i], sort_custo1[i]);
		}
		printf("\n Custo 2 \n");
		for (i = 0; i < n_pop; i++){
			printf("%d|%f  ", sort_chaves2[i], old_custo2[i]);
		}
		printf("\n");
		for (i = 0; i < n_pop; i++){
			printf("%d|%f  ", sort_chaves2[i], sort_custo2[i]);
		}
	}

	if (individuo == sort_chaves1[index_inicio] || individuo == sort_chaves2[index_inicio] || individuo == sort_chaves1[index_fim] || individuo == sort_chaves2[index_fim]){
		sh_pop_dist[tid] = 99999;
	}
	else{
		for (i = 0; i < n_pop; i++){
			if (sort_chaves1[i] == individuo){
				pos_c1 = i;
				break;
			}
		}
		for (i = 0; i < n_pop; i++){
			if (sort_chaves2[i] == individuo){
				pos_c2 = i;
				break;
			}
		}

		//printf("\n\n tid %d individuo %d pos1 %d pos2 %d\n C1| pos+1 %f pos-1 %f\n C2| pos+1 %f pos-1 %f\n", tid, individuo, pos_c1, pos_c2, sort_custo1[pos_c1 + 1], sort_custo1[pos_c1 - 1], sort_custo2[pos_c2 + 1], sort_custo2[pos_c2 - 1]);

		sh_pop_dist[tid] = (sort_custo1[pos_c1+1] - sort_custo1[pos_c1-1])/(sort_custo1[index_fim] - sort_custo1[index_inicio]);
		sh_pop_dist[tid] += (sort_custo2[pos_c2 + 1] - sort_custo2[pos_c2 - 1]) / (sort_custo2[index_fim] - sort_custo2[index_inicio]);
	}

	pop_dist[individuo] = sh_pop_dist[tid];
	printf("multidao fim");
	/*
	if (tid == 0){
		printf("\n Multidao \n");
		for (i = 0; i < n_pop; i++){
			printf("%d|%f\n", i, multidao[i]);
		}
	}
	*/
}

__global__ void init_rand_selecao(curandState* globalState, int n_pop, unsigned int *rand_selecao){
	int tid = threadIdx.x;	
	curandState localState = globalState[tid];
	float random = curand_uniform(&localState);
	rand_selecao[tid] = (int)(random * n_pop);
	globalState[tid] = localState;

	//printf("\n(%d)random %d",tid ,rand_selecao[tid]);
}

__global__ void init_rand_cruzamento(curandState* globalState, float *rand_cruzamento){
	int tid = threadIdx.x;
	curandState localState = globalState[tid];
	float random = curand_uniform(&localState);
	rand_cruzamento[tid] = random;
	globalState[tid] = localState;

	//printf("\n(%d)random %d",tid ,rand_selecao[tid]);
}

__global__ void init_rand_mutacao(curandState* globalState, float *rand_mutacao){
	int tid = threadIdx.x;
	curandState localState = globalState[tid];
	float random = curand_uniform(&localState);
	rand_mutacao[tid] = random;
	globalState[tid] = localState;

	//printf("\n(%d)random %f", tid, rand_mutacao[tid]);
}

__global__ void init_rand_pos_mutacao(curandState* globalState, unsigned int *rand_pos_mutacao){
	int tid = threadIdx.x;
	curandState localState = globalState[tid];
	float random = curand_uniform(&localState);
	rand_pos_mutacao[tid] = (int)(random * 6);
	globalState[tid] = localState;

	//printf("\n(%d)random %d",tid ,rand_selecao[tid]);
}

__global__ void selecao_cruzamento_mutacao(unsigned int n_pop, unsigned int n_filhos,float *pop_pos, unsigned int *pop_rank,
										   float *pop_dist, unsigned int *rand_selecao, float *rand_cruzamento, float *rand_mutacao,
										   unsigned int *rand_pos_mutacao, float p_mutacao){
	int tid = threadIdx.x;

	__shared__ unsigned int sh_rank[6];
	__shared__ float sh_pos[6*6];
	__shared__ float sh_multidao[6];

	float filho1[6];
	float filho2[6];

	unsigned int dif_threads = n_pop - n_filhos;

	if (tid < dif_threads){

		sh_pos[((n_filhos + tid) * 6)] = pop_pos[((n_filhos + tid) * 6)];
		sh_pos[((n_filhos + tid) * 6) + 1] = pop_pos[((n_filhos + tid) * 6) + 1];
		sh_pos[((n_filhos + tid) * 6) + 2] = pop_pos[((n_filhos + tid) * 6) + 2];
		sh_pos[((n_filhos + tid) * 6) + 3] = pop_pos[((n_filhos + tid) * 6) + 3];
		sh_pos[((n_filhos + tid) * 6) + 4] = pop_pos[((n_filhos + tid) * 6) + 4];
		sh_pos[((n_filhos + tid) * 6) + 5] = pop_pos[((n_filhos + tid) * 6) + 5];

		sh_multidao[n_filhos + tid] = pop_dist[n_filhos + tid];
		sh_rank[n_filhos + tid] = pop_rank[n_filhos + tid];

		/*
		printf("in%d - %d\n",tid ,(n_filhos / 2) + tid);
		sh_pos[(((n_filhos/2)+tid)*6)] = pop_pos[(((n_filhos/2)+tid)*6)];
		sh_pos[(((n_filhos/2)+tid)*6)+1] = pop_pos[(((n_filhos/2)+tid)*6)+1];
		sh_pos[(((n_filhos/2)+tid)*6)+2] = pop_pos[(((n_filhos/2)+tid)*6)+2];
		sh_pos[(((n_filhos/2)+tid)*6)+3] = pop_pos[(((n_filhos/2)+tid)*6)+3];
		sh_pos[(((n_filhos/2)+tid)*6)+4] = pop_pos[(((n_filhos/2)+tid)*6)+4];
		sh_pos[(((n_filhos/2)+tid)*6)+5] = pop_pos[(((n_filhos/2)+tid)*6)+5];

		sh_multidao[(n_filhos/2)+tid] = pop_dist[(n_filhos/2)+tid];
		sh_rank[(n_filhos/2)+tid] = pop_rank[(n_filhos/2)+tid];

		sh_pos[(((n_filhos/2)+tid)*12)] = pop_pos[(((n_filhos/2)+tid)*12)];
		sh_pos[(((n_filhos/2)+tid)*12)+1] = pop_pos[(((n_filhos/2)+tid)*12)+1];
		sh_pos[(((n_filhos/2)+tid)*12)+2] = pop_pos[(((n_filhos/2)+tid)*12)+2];
		sh_pos[(((n_filhos/2)+tid)*12)+3] = pop_pos[(((n_filhos/2)+tid)*12)+3];
		sh_pos[(((n_filhos/2)+tid)*12)+4] = pop_pos[(((n_filhos/2)+tid)*12)+4];
		sh_pos[(((n_filhos/2)+tid)*12)+5] = pop_pos[(((n_filhos/2)+tid)*12)+5];

		sh_multidao[((n_filhos/2)+tid)*2] = pop_dist[((n_filhos/2)+tid)*2];
		sh_rank[((n_filhos/2)+tid)*2] = pop_rank[((n_filhos/2)+tid)*2];
		*/
	}
	sh_pos[(tid*6)] = pop_pos[(tid*6)];
	sh_pos[(tid*6)+1] = pop_pos[(tid*6)+1];
	sh_pos[(tid*6)+2] = pop_pos[(tid*6)+2];
	sh_pos[(tid*6)+3] = pop_pos[(tid*6)+3];
	sh_pos[(tid*6)+4] = pop_pos[(tid*6)+4];
	sh_pos[(tid*6)+5] = pop_pos[(tid*6)+5];

	sh_multidao[tid] = pop_dist[tid];
	sh_rank[tid] = pop_rank[tid];

	__syncthreads();
	/*
	if (tid == 0){
		for (int i = 0; i < n_pop; i++){
			printf("\nGlobal (%d) %f %f %f %f %f %f",i, pop_pos[(i * 6)], pop_pos[(i * 6) + 1], pop_pos[(i * 6) + 2], pop_pos[(i * 6) + 3], pop_pos[(i * 6) + 4], pop_pos[(i * 6) + 5]);
			printf("\nShared (%d) %f %f %f %f %f %f",i, sh_pos[(i * 6)], sh_pos[(i * 6) + 1], sh_pos[(i * 6) + 2], sh_pos[(i * 6) + 3], sh_pos[(i * 6) + 4], sh_pos[(i * 6) + 5]);

			printf("\nMultidao GL(%d) %f", i, pop_dist[i]);
			printf("\nMultidao SH(%d) %f", i, sh_multidao[i]);
		}
	}
	*/
	int i = rand_selecao[(tid * 4)];
	
	unsigned int candidato_1 = rand_selecao[(tid * 4)];	
	unsigned int candidato_2 = rand_selecao[(tid * 4) + 1];

	while (candidato_1 == candidato_2){
		candidato_2 = rand_selecao[i];
		i++;
	}

	unsigned int candidato_3 = rand_selecao[(tid * 4) + 2];
	unsigned int candidato_4 = rand_selecao[(tid * 4) + 3];

	while (candidato_3 == candidato_4){
		candidato_4 = rand_selecao[i];
		i++;
	}

	unsigned int pai_1;
	unsigned int pai_2;
	
	//printf("\nTID %d | %d %d %d %d", tid, candidato_1, candidato_2, candidato_3, candidato_4);

	if (sh_rank[candidato_1] < sh_rank[candidato_2]){
		pai_1 = candidato_1;
	}
	else if (sh_rank[candidato_2] < sh_rank[candidato_1]){
		pai_1 = candidato_2;
	}
	else{
		if (sh_multidao[candidato_1] > sh_multidao[candidato_2]){
			pai_1 = candidato_1;
		}
		else{
			pai_1 = candidato_2;
		}
	}

	if (sh_rank[candidato_3] < sh_rank[candidato_4]){
		pai_2 = candidato_3;
	}
	else if (sh_rank[candidato_4] < sh_rank[candidato_3]){
		pai_2 = candidato_4;
	}
	else{
		if (sh_multidao[candidato_3] > sh_multidao[candidato_4]){
			pai_2 = candidato_3;
		}
		else{
			pai_2 = candidato_4;
		}
	}
	
	for (i = 0; i < 6; i++){
		filho1[i] = (sh_pos[(pai_1*6)+i] * (rand_cruzamento[(tid*6)+i])) + (sh_pos[(pai_2*6)+i]*(1-rand_cruzamento[(tid*6)+i]));
		filho2[i] = (sh_pos[(pai_2*6)+i] * (rand_cruzamento[(tid*6)+i])) + (sh_pos[(pai_1*6)+i]*(1-rand_cruzamento[(tid*6)+i]));
	}
	/*
	printf("\n\nTID(%d) %f %f %f %f %f %f\nPai 1 %f %f %f %f %f %f\nPai 2 %f %f %f %f %f %f\nFilho 1 : %f %f %f %f %f %f \nFilho 2 : %f %f %f %f %f %f ",
		tid, rand_cruzamento[(tid * 6)], rand_cruzamento[(tid * 6) + 1], rand_cruzamento[(tid * 6) + 2], rand_cruzamento[(tid * 6) + 3], rand_cruzamento[(tid * 6) + 4], rand_cruzamento[(tid * 6) + 5],
		sh_pos[(pai_1 * 6)], sh_pos[(pai_1 * 6) + 1], sh_pos[(pai_1 * 6) + 2], sh_pos[(pai_1 * 6) + 3], sh_pos[(pai_1 * 6) + 4], sh_pos[(pai_1 * 6) + 5],
		sh_pos[(pai_2 * 6)], sh_pos[(pai_2 * 6) + 1], sh_pos[(pai_2 * 6) + 2], sh_pos[(pai_2 * 6) + 3], sh_pos[(pai_2 * 6) + 4], sh_pos[(pai_2 * 6) + 5],
		filho1[0], filho1[1], filho1[2], filho1[3], filho1[4], filho1[5],
		filho2[0], filho2[1], filho2[2], filho2[3], filho2[4], filho2[5]);
	*/
	float local_rand_mutacao1 = rand_mutacao[(tid*2)];
	float local_rand_mutacao2 = rand_mutacao[(tid*2)+1];

	if (local_rand_mutacao1 <= p_mutacao){
		filho1[rand_pos_mutacao[(tid*2)]] = 70 + local_rand_mutacao1 * 70;
	}
	if (local_rand_mutacao2 <= p_mutacao){
		filho2[rand_pos_mutacao[(tid*2)+1]] = 70 + local_rand_mutacao2 * 70;
	}


	printf("\n\nTID(%d) \nFilho 1 : %f %f %f %f %f %f \nFilho 2 : %f %f %f %f %f %f ",
		tid,
		filho1[0], filho1[1], filho1[2], filho1[3], filho1[4], filho1[5],
		filho2[0], filho2[1], filho2[2], filho2[3], filho2[4], filho2[5]);

	if (tid == 0){
		for (i = 0; i < n_pop + n_filhos; i++){
			printf("\n(%d)A %f %f %f %f %f %f", i, pop_pos[(i*6)], pop_pos[(i*6)+1], pop_pos[(i*6)+2], pop_pos[(i*6)+3], pop_pos[(i*6)+4], pop_pos[(i*6)+5]);
		}
	}

	pop_pos[(n_pop+tid)*6] = filho2[0];
	pop_pos[(n_pop+tid)*6+1] = filho2[1];
	pop_pos[(n_pop+tid)*6+2] = filho2[2];
	pop_pos[(n_pop+tid)*6+3] = filho2[3];
	pop_pos[(n_pop+tid)*6+4] = filho2[4];
	pop_pos[(n_pop+tid)*6+5] = filho2[5];

	printf("\n\n");
	if (tid == 0){
		for (i = 0; i < n_pop + n_filhos; i++){
			printf("\n(%d)B %f %f %f %f %f %f", i, pop_pos[(i * 6)], pop_pos[(i * 6) + 1], pop_pos[(i * 6) + 2], pop_pos[(i * 6) + 3], pop_pos[(i * 6) + 4], pop_pos[(i * 6) + 5]);
		}
	}
}

__global__ void selecao(int n_pop, unsigned int *pop_rank, float *pop_dist, unsigned int *tam_fronts,
								unsigned int *fronts, float *sort_custo1, unsigned int *sort_chaves1){
	unsigned int tid = threadIdx.x;

	if (tid == 0){
		printf("\n\nFronteira %d %d %d %d %d %d %d %d %d %d", fronts[0], fronts[1], fronts[2], fronts[3], fronts[4], fronts[5], fronts[6], fronts[7], fronts[8], fronts[9]);
		printf("\nTam Fronts %d %d %d %d %d %d %d %d %d %d", tam_fronts[0], tam_fronts[1], tam_fronts[2], tam_fronts[3], tam_fronts[4], tam_fronts[5], tam_fronts[6], tam_fronts[7], tam_fronts[8], tam_fronts[9]);
		printf("\nDist %f %f %f %f %f %f %f %f %f %f", pop_dist[fronts[0]], pop_dist[fronts[1]], pop_dist[fronts[2]], pop_dist[fronts[3]], pop_dist[fronts[4]], pop_dist[fronts[5]], pop_dist[fronts[6]], pop_dist[fronts[7]], pop_dist[fronts[8]], pop_dist[fronts[9]]);
	}

	__shared__ unsigned int sh_fronts[10];
	__shared__ unsigned int sh_tam_fronts[10];
	__shared__ float sh_pop_dist[10];

	unsigned int index_inicio = 0;
	unsigned int index_fim;
	unsigned int individuo;
	unsigned int local_rank;
	unsigned int pos;

	sh_fronts[tid] = fronts[tid];
	sh_tam_fronts[tid] = tam_fronts[tid];
	individuo = sh_fronts[tid];
	sort_chaves1[tid] = sh_fronts[tid];
	sort_custo1[tid] = pop_dist[individuo];

	local_rank = pop_rank[individuo];
	int i = 0;

	while (i + 1 != local_rank){
		index_inicio += sh_tam_fronts[i];
		i++;
	}
	index_fim = index_inicio + sh_tam_fronts[local_rank - 1] - 1;
		
	if (individuo == sh_fronts[index_inicio] && sh_tam_fronts[local_rank - 1] > 1){
		printf("\n individuo %d rank %d index %d",individuo, local_rank,index_inicio);
		bool impar = (bool)(sh_tam_fronts[local_rank - 1] % 2);
		int tam_sort = sh_tam_fronts[local_rank - 1];

		if (impar){
			tam_sort++;
		}
		bitonic_sort<<<1, tam_sort, (sizeof(float)*tam_sort * 2) >>>(&sort_custo1[index_inicio], &sort_chaves1[index_inicio], tam_sort, impar);

		for (int i = 0; i < sh_tam_fronts[local_rank - 1];i++){
			fronts[index_inicio + i] = sort_chaves1[index_inicio + i];
		}
	}

	if (tid == 0){
		printf("\n\nFronteira %d %d %d %d %d %d %d %d %d %d", fronts[0], fronts[1], fronts[2], fronts[3], fronts[4], fronts[5], fronts[6], fronts[7], fronts[8], fronts[9]);
		//printf("\nDist %f %f %f %f %f %f %f %f %f %f", pop_dist[fronts[0]], pop_dist[fronts[1]], pop_dist[fronts[2]], pop_dist[fronts[3]], pop_dist[fronts[4]], pop_dist[fronts[5]], pop_dist[fronts[6]], pop_dist[fronts[7]], pop_dist[fronts[8]], pop_dist[fronts[9]]);
	}
}

int main(int argc, char *argv[])
{
	curandState* devStates;

	float p_cruzamento = 0.8;
	float p_mutacao = 0.02;

	unsigned int n_pop = 6;
	unsigned int n_filhos = 4; // n_filhos = roundf(n_pop*p_cruzamento/2)*2;
	unsigned int n_pop_total = n_pop + n_filhos;
	unsigned int n_var = n_pop_total * 6;
	unsigned int n_custo = n_pop_total * 2;
	unsigned int n_conj_domi = n_pop_total * n_pop_total;
	
	float* pop_pos;
	float* pop_custo;
	float* pop_dist;
	float* sort_custo1;
	float* sort_custo2;
	float* rand_mutacao;
	float* rand_cruzamento;
	
	bool* conj_domi;

	unsigned int* sort_chaves1;
	unsigned int* sort_chaves2;
	unsigned int* ctd_domi;
	unsigned int* pop_rank;
	unsigned int* fronts;
	unsigned int* index_front;
	unsigned int* tam_fronts;
	unsigned int* rand_selecao;	
	unsigned int* rand_pos_mutacao;

	float* pop0_teste;
	pop0_teste = (float *)malloc(n_var);

	CUDA_CALL(cudaMalloc((void **)&devStates, n_var*sizeof(curandState)));

	CUDA_CALL(cudaMalloc((void **)&pop_pos, n_var*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&pop_custo, n_custo*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&pop_dist, n_pop_total*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&sort_custo1, n_pop_total*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&sort_custo2, n_pop_total*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&rand_cruzamento, 6*n_filhos*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&rand_mutacao, 2 * n_filhos*sizeof(float)));

	CUDA_CALL(cudaMalloc((void **)&ctd_domi, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&pop_rank, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&tam_fronts, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&fronts, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&index_front, sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&sort_chaves1, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&sort_chaves2, n_pop_total*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&rand_selecao, n_filhos*sizeof(unsigned int)));	
	CUDA_CALL(cudaMalloc((void **)&rand_pos_mutacao, 2*n_filhos*sizeof(unsigned int)));

	CUDA_CALL(cudaMalloc((void **)&conj_domi, n_conj_domi*sizeof(bool)));

	setup_rand <<< 1, n_var >>> (devStates, time(NULL));
	gen_pop0 <<< 1, n_var >>> (devStates, pop_pos);
	//individuo_teste <<<1, 1 >>> (pop0_pos);

	init_rand_selecao<<< 1, 4 * n_filhos >>>(devStates, n_pop, rand_selecao);
	init_rand_cruzamento<<<1, 6 * n_filhos >>>(devStates, rand_cruzamento);
	init_rand_mutacao<<< 1, 2 * n_filhos >>>(devStates, rand_mutacao);
	init_rand_pos_mutacao<<< 1, 2 * n_filhos >>>(devStates, rand_pos_mutacao);

	custo <<< 1, n_pop >>>(pop_pos, pop_custo,0);
	init_fronts<<< 1, n_pop >>>(tam_fronts, fronts);

	NDS1<<< 1, n_pop >>>(pop_pos, pop_custo, ctd_domi, conj_domi, pop_rank, n_pop, tam_fronts, fronts, index_front);
	NDS2_pai<<< 1, 1 >>>(n_pop, pop_pos, conj_domi,ctd_domi, pop_rank, tam_fronts, fronts,index_front);
	dist_multidao<<< 1, n_pop >>>(n_pop, pop_rank, pop_custo, pop_dist, tam_fronts, fronts, sort_custo1, sort_custo2, sort_chaves1, sort_chaves2);
	cudaDeviceSynchronize();
	selecao_cruzamento_mutacao<<< 1, n_filhos >>>(n_pop, n_filhos, pop_pos, pop_rank, pop_dist, rand_selecao,rand_cruzamento,rand_mutacao,rand_pos_mutacao,p_mutacao);

	custo<<< 1,n_filhos >>>(pop_pos,pop_custo,n_pop);
	init_fronts<<< 1, n_pop_total >>>(tam_fronts, fronts);
	NDS1<<< 1, n_pop_total >>>(pop_pos, pop_custo, ctd_domi, conj_domi, pop_rank, n_pop_total, tam_fronts, fronts, index_front);
	NDS2_pai<<< 1, 1 >>>(n_pop_total, pop_pos, conj_domi,ctd_domi, pop_rank, tam_fronts, fronts,index_front);
	dist_multidao<<< 1, n_pop_total >>>(n_pop_total, pop_rank, pop_custo, pop_dist, tam_fronts, fronts, sort_custo1, sort_custo2, sort_chaves1, sort_chaves2);
	selecao<<< 1, n_pop_total >>>(n_pop_total, pop_rank, pop_dist, tam_fronts, fronts, sort_custo1, sort_chaves1);

	cudaDeviceSynchronize();
	CUDA_CALL(cudaFree(pop_custo));
	CUDA_CALL(cudaFree(pop_pos));
	free(pop0_teste);
	
	return 0;
}