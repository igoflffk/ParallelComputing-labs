#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
void combSort(double arr[], int n) {
	int gap = n;
	bool swapped = true;
	int temp;
	while (gap != 1 || swapped) {
		gap *= 10 / 13;
		if (gap < 1) {
			gap = 1;
		}
		swapped = false;
		for (int i = 0; i < n - gap; i++) {
			if (arr[i] > arr[i + gap]) {
				temp = arr[i];
				arr[i] = arr[i + gap];
				arr[i + gap] = temp;
				swapped = true;
			}
		}
	}
}
int main(int argc, char* argv[])
{
	int i, j, N;
	struct timeval T1, T2;
	double e = exp(1.0);
	double min = 1;
	double max = 648;
	long delta_ms;
	// N равен первому параметру командной строки
	N = atoi(argv[1]);
	double* restrict M1 = (double*)malloc(N * sizeof(double));
	double* restrict M2 = (double*)malloc(N / 2 * sizeof(double));
	double* restrict M2_copy = (double*)malloc(N / 2 * sizeof(double));
	unsigned int seed = 1;

	// запомнить текущее время T1
	gettimeofday(&T1, NULL);
	for (i = 0; i < 100; i++) // 100 экспериментов
	{
		seed = i; // инициализировать начальное значение ГСЧ
		for (j = 0; j < N; j++) // Заполнить массив исходных данных размером N
		{
			M1[j] = (((double)rand_r(&seed) / (RAND_MAX)) * (max - min) + min);
		}
		for (j = 0; j < N / 2; j++)
		{
			M2[j] = ((double)rand_r(&seed) / (RAND_MAX)) * (max * 10 - max) + max;
			M2_copy[j] = M2[j];
		}
		6
			//map
			for (j = 0; j < N; j++)
			{
				// Гиперболический тангенс с последующим уменьшением на 1
				M1[j] = ((tanh(M1[j])) - 1);
			}
		for (j = 1; j < N / 2; j++)
		{
			// Десятичный логарифм, возведенный в степень e 
			M2[j] = pow((M2[j] + M2_copy[j - 1]), e);
		}
		M2[0] = pow(M2[0], e);
		for (j = 0; j < N / 2; j++) // Решить поставленную задачу, заполнить массив 
			с результатами
		{
			//Деление (т.е. M2[i] = M1[i]/M2[i])
			M2[j] = M1[j] / M2[j];
		}

			//combSort(M2, N / 2); 
			// Сортировка расчёской (Comb sort) результатов
			//минимальные не нулевой на два фора 
		double min_nonzero = M2[0];
		double sum_sin = 0.0;
		for (j = 0; j < N / 2; j++)
		{
			if ((M2[j] > 0) && (M2[j] < min_nonzero))
			{
				min_nonzero = M2[j];
			}
		}
		for (j = 0; j < N / 2; j++)
		{
			if ((int)(M2[j] / min_nonzero) % 2 == 0)
			{
				sum_sin += sin(M2[j]);
			}
		}
		printf("%.10lf\n", sum_sin);
	}
	gettimeofday(&T2, NULL); // запомнить текущее время T2
	delta_ms = (T2.tv_sec - T1.tv_sec) * 1000 + (T2.tv_usec - T1.tv_usec) / 1000;
	printf("\nN=%d. Milliseconds passed: %ld\n", N, delta_ms);
	free(M1);
	free(M2);
	free(M2_copy);
	return 0;
}