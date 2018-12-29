
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//#include "Filterpy_UNIT_AlgorithmDetect.h"

/** 
 * constants 
 */
// # Percentage threshold on anomaly score below which is considered noises.
DEFAULT_NOISE_PCT_THRESHOLD = 0.001

// Private Var
float gfMemScore[1664];
float gfMemDerivatives[1664];
float gfMemData[1664];
float gfMemMea[1664];

/**
 * Compute derivatives of the time series.
 */
void unit_detect_compute_derivatives(void)
{
    unsigned short i = 0;
    unsigned short td = 1; // 求导数的分母
    float derivative;

    for(i=1; i< 1664; i++){ // 跳过第1个点        
        derivative = (gfMemData[i] - gfMemData[i-1]) / td;
        derivative = abs(derivative);
        gfMemDerivatives[i] = derivative;
    }
    
    //# First timestamp is assigned the same derivative as the second timestamp.
    gfMemDerivatives[0] = 0;
}

/**
 * Compute exponential moving average of a list of points.
 * :param float smoothing_factor: the smoothing factor.
 * :param list points: the data points.
 * :return list: all ema in a list.
 */
void unit_detect_compute_ema(float smoothing_factor, float* derivatives, unsigned short n_len)
{
    unsigned short i = 0;

    memset(&gfMemMea[0], 0, sizeof(gfMemMea));
    //# The initial point has a ema equal to itself.
    gfMemMea[0] = derivatives[0];
    for(i=1; i<n_len; i++){
        gfMemMea[i] = smoothing_factor * derivatives[i] + (1 - smoothing_factor) * gfMemMea[i-1];
    }   
}

void unit_detect_denoise_scores(void)
{

}

/**
 * Compute anomaly scores for the time series.
 */
void unit_detect_set_scores(void)
{
    unsigned short i = 0;
    float f_derivatives_ema;
    float anom_scores[1664], derivatives_ema[1664];
    
    memeset(&anom_scores[0], 0, sizeof(anom_scores));
    unit_detect_compute_derivatives();
    //f_derivatives_ema = utils.compute_ema(self.smoothing_factor, self.derivatives);
    derivatives_ema = unit_detect_compute_ema(gf_smoothing_factor, gfMemDerivatives, 1664);
    for(i=0; i<1664; i++){
        anom_scores[i] = abs(gfMemDerivatives[i] - derivatives_ema[i]);
    }
}

/**
 * 
 */
void main(){
    unit_detect_compute_ema(0, 0, 1664);
    return;
}
	
