#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//#include "Filterpy_UNIT_AlgorithmDetect.h"

// Constants var  &  Macro
// 128*13=1664, Total 13 Cycels
#define UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM 1664
#define UNIT_DETECT_PER_FAULTWAVE_FRAME_NUM 128
#define UNIT_DETECT_CORRELATOR_THRESHOLD 0.7

// private aabs |...|
#define aabs(a) (a)>0?(a):(0-(a))

// # Percentage threshold on anomaly score below which is considered noises.
#define DEFAULT_NOISE_PCT_THRESHOLD 0.001
// # Constants for DerivativeDetector.
#define DEFAULT_DERI_SMOOTHING_FACTOR 0.2
float gf_smoothing_factor = 0.2;  //DEFAULT_DERI_SMOOTHING_FACTOR;

// Private Var
static float gfMemScore[UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM];
static float gfMemDerivatives[UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM];
static float gfMemData[UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM];

// Record the MAX-Pointer data Information.
typedef struct
{
    float fData;     			/**< Value of the MAX-Pointer data.     */
    unsigned short nIndex;      /**< Index of the MAX-Pointer data.     */    
} INDEX_F32_STRUCT;
INDEX_F32_STRUCT stCycleAll, stCycleFstSecThd, stCycleFourFive;

// Record the Range of continuous MAX-Pointer Information.
#define UNIT_DETECT_RANGE_INDEX_NUM 3
typedef struct
{
    float anomaly_score;            /**< Sum-Score of the MAX-Pointer Range . */
    unsigned short index_start;     /**< Start Index of the MAX-Pointer Range . */
    unsigned short index_end;       /**< End Index of the MAX-Pointer Range . */
} RANGE_INDEX_F32_STRUCT;
RANGE_INDEX_F32_STRUCT stRangeIndex[UNIT_DETECT_RANGE_INDEX_NUM]; // Only Record 'UNIT_DETECT_RANGE_INDEX_NUM' Array.

// Func declaration.
static float unit_detect_algorithm_correlator(unsigned short nIndex, float *p_data, short n_len);
static unsigned short unit_detect_algorithm_anomalies_range(float *p_data, short n_len);

/**
 * Compute derivatives of the time series.
 */
void unit_detect_compute_derivatives(void)
{
    unsigned short i = 0;
    float td = 2000.0;//1; // 求导数的分母
    float derivative = 0.0;

    for(i=1; i< UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM; i++){      
        gfMemDerivatives[i] = (gfMemData[i] - gfMemData[i-1]) / td;
        gfMemDerivatives[i] = aabs(gfMemDerivatives[i]);
        //gfMemDerivatives[i] = fabsf(gfMemDerivatives[i]);
        //printf("%f \r\n", gfMemDerivatives[i]); //#debug
        //printf("gfMemDerivatives[%d] \t=\t %f \r\n", i, gfMemDerivatives[i]); //#debug
    }
    
    //# First timestamp is assigned the same derivative as the second timestamp.
    gfMemDerivatives[0] = gfMemDerivatives[1];
}

/**
 * Compute exponential moving average of a list of points.
 * :param float smoothing_factor: the smoothing factor.
 * :param list points: the data points.
 * :return list: all ema in a list.
 */
void unit_detect_compute_ema(float smoothing_factor, float* derivatives, unsigned short n_len, float* derivatives_mea)
{
    unsigned short i = 0;

    memset(derivatives_mea, 0, n_len*sizeof(float));
    //# The initial point has a ema equal to itself.
    *(derivatives_mea) = derivatives[0];
    for(i=1; i<n_len; i++){
        *(derivatives_mea + i) = smoothing_factor * derivatives[i] + (1 - smoothing_factor) * (*(derivatives_mea + i-1));
    }
}


/**
 * 去噪声，暂时无用。
 * ----------
 */
void unit_detect_denoise_scores(void)
{

}

/**
 * Compute ALL-STD-StandardDeviation, 总体标准偏差。
 * ----------
 * Parameters
 * p_anom_scores_values : float*
 *   U0 Data .
 * n_len : unsigned short
 *   Len of U0 Data .
 */
float unit_detect_calc_STD(float* p_anom_scores_values, unsigned short n_len)
{
    unsigned short i = 0;
    float fSum=0.0, fAvg=0.0, fSpow=0.0;
    if( (NULL == p_anom_scores_values) || (0 == n_len) ) {
        return 0; // Fetal Error
    }
    for(i=0; i<n_len; i++){
        fSum += *(p_anom_scores_values+i);
        //printf("%f \r\n", *(p_anom_scores_values+i));
    }
    //printf("fSum == %f \r\n", fSum);
    fAvg = fSum/n_len;
    //printf("fAvg == %f \r\n", fAvg);
    //for(i=0; i<n_len+1; i++){
    for(i=0; i<n_len; i++){
        fSpow += (*(p_anom_scores_values+i) - fAvg) * (*(p_anom_scores_values+i) - fAvg);
    }

    return (sqrt(fSpow/n_len));
}

/**
 * Compute anomaly scores for the time series.
 * ----------
 * 
 * anom_scores : float*
 *   Need to be == gfMemScore[]
 */
void unit_detect_set_scores(float* anom_scores)
{
    unsigned short i = 0;
    float f_derivatives_ema;
    float /*anom_scores[1664],*/ derivatives_ema[UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM];
    float fStdev = 0.0;
    
    //memset(&anom_scores[0], 0, UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM*sizeof(float));
    memset(&stCycleAll, 0, sizeof(stCycleAll));
    memset(&stCycleFstSecThd, 0, sizeof(stCycleFstSecThd));
    memset(&stCycleFourFive, 0, sizeof(stCycleFourFive));

    unit_detect_compute_derivatives();
    // f_derivatives_ema = utils.compute_ema(self.smoothing_factor, self.derivatives);
    unit_detect_compute_ema(gf_smoothing_factor, gfMemDerivatives, UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM, derivatives_ema);
    for(i=0; i<UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM; i++){
        anom_scores[i] = fabsf(gfMemDerivatives[i] - derivatives_ema[i]);
        //printf("%d \t %f \t=\t %f\t -\t %f \r\n", i, anom_scores[i], gfMemDerivatives[i], derivatives_ema[i]); 
    }
    // Calc STD ===>  STDEV.P(A1:A1664)
    // stdev = numpy.std(list(anom_scores.values())) // anom_scores[i]
    fStdev = unit_detect_calc_STD( anom_scores, UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM);
    //printf("%f \r\n", fStdev);
    if(fStdev){
        // fStdev
        for(i=0; i<=383; i++){
            anom_scores[i] /= fStdev;
            //gfMemScore[i] = anom_scores[i];             
            // Record the ALL-Cycle MAX-Point.
            if(anom_scores[i] > stCycleFstSecThd.fData){
                stCycleFstSecThd.nIndex = i;
                stCycleFstSecThd.fData = anom_scores[i];
            }
        }
        //printf("stCycleFstSecThd.nIndex = %d \t stCycleFstSecThd.fData = %f \r\n", stCycleFstSecThd.nIndex, stCycleFstSecThd.fData);

        // (2) 384~639, 第4~5周波
        stCycleFourFive.nIndex = stCycleFstSecThd.nIndex;
        stCycleFourFive.fData = stCycleFstSecThd.fData;
        for(i=384; i<=639; i++){
            anom_scores[i] /= fStdev;
            if(anom_scores[i] > stCycleFourFive.fData){
                stCycleFourFive.nIndex = i;
                stCycleFourFive.fData = anom_scores[i];
            }
        }
        //printf("stCycleFourFive.nIndex = %d \t stCycleFourFive.fData = %f \r\n", stCycleFourFive.nIndex, stCycleFourFive.fData);

        // (3) 640~1664, 第6~13周波
        stCycleAll.nIndex = stCycleFourFive.nIndex;
        stCycleAll.fData = stCycleFourFive.fData;
        for(i=640; i<UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM; i++){
            anom_scores[i] /= fStdev;
            if(anom_scores[i] > stCycleAll.fData){
                stCycleAll.nIndex = i;
                stCycleAll.fData = anom_scores[i];
            }
        }
        //printf("stCycleAll.nIndex = %d \t stCycleAll.fData = %f \r\n", stCycleAll.nIndex, stCycleAll.fData);
    }
}

/**
 * Find the MAX-Abnormal-Point.
 * ----------
 * Return： the Index of SuddenChange-U0 .
 */
unsigned short unit_detect_anomalies(void)
{
    unsigned short i=0, j=0;
    
    if(383 > stCycleAll.nIndex){
        // 1. MAX-Score 不应该落在前 3 个周波内，所以 Fetal error .
        return 0; 
    }else if(384<=stCycleAll.nIndex  && stCycleAll.nIndex<=639){
        // 2. ××× 如果 MAX-Score 存在在第4~5周波内，则该MAX-Score就是U0的故障起点。 
        //    ××× @2019-01-09 验证波形 SAR-device.sdb.await__PDZ810_20190108__981_BAY01_0219_20181126_060056_456__U0 时此种情况不成立。
        //    ×××  所以必须逐个点判别了。
        //return stCycleAll.nIndex;
        // 2.1 +++ @2019-01-09 优先寻找 3 个连续的值大于 1.2 倍 stCycleFstSecThd.fData
        for(i=384; i<=stCycleAll.nIndex; i++){            
            unsigned short nOverChangeAvg = 0;   // 累计平均值，对应 j 值
            unsigned short nOverChangeFirst = 0; // 当前值的 Score 大于 1.2 倍  
            float f_local_sum = 0;  // 计算平均值的临时变量

            // 2.1.1 如果 stCycleFstSecThd.fData=0, 则说明前三个周波的采样点很平滑且无波动, 所以需要先找到一个 !=0 的导数值。
            if(0==gfMemScore[i]){
                continue;
            }
            if( (0==stCycleFstSecThd.fData) && (gfMemScore[i]!=0) ){
                //stCycleFstSecThd.fData = gfMemScore[i];
                return i; // 返回第一个非0的导数值。
            }//[End] 2.1.1 

            if(gfMemScore[i] > 1.2*stCycleFstSecThd.fData){
                nOverChangeFirst = 1;
                for(j=0; j<3; j++){
                    f_local_sum += gfMemScore[i+j-1];                
                }
                if(f_local_sum/3 > 1.1*stCycleFstSecThd.fData){
                    nOverChangeAvg = 1;
                }
            }

            if((nOverChangeAvg >= 1) && (gfMemScore[i-1] <= gfMemScore[i])){
                // When nOverChange==3, find the MAX-Point, otherwise nOverChange<3 .
                // and [i] <= [i+1]
                // If find the MAX-Point, return the Index of U0[i].
                //break;
                return i;
            }else if(nOverChangeFirst >= 1){
                // 仅更新待比较的 Score 值 @2019-01-13 不再更新 score，则仍用 1~3 周波内的最大 score
                //stCycleFstSecThd.fData = gfMemScore[i];
            }

            nOverChangeAvg = 0; // 清零，准备下一次的 j 循环。
            nOverChangeFirst = 0;
        }
    }else if(640<=stCycleAll.nIndex && stCycleAll.nIndex<UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM){
        // DEL---3. 如果，在 13 个周波内，MAX-Score 不在第4~5周波内，且出现在第5个周波之后。 ---DEL
        // DEL---   优先寻找 3 个连续的值大于 1.2 倍 stCycleFstSecThd.fData                ---DEL
        //for(i=384; i<(640+128); i++){ //@2019-01-12 11:00 +128 
        for(i=384; i<=stCycleAll.nIndex; i++){  //@2019-01-12 16:58 
            unsigned short nOverChangeAvg = 0;   // 累计平均值，对应 j 值
            unsigned short nOverChangeFirst = 0; // 当前值的 Score 大于 1.2 倍            
            float f_local_sum = 0;  // 计算平均值的临时变量

            // 3.1.1 如果 stCycleFstSecThd.fData=0, 则说明前三个周波的采样点很平滑且无波动, 所以需要先找到一个 !=0 的导数值。
            if(0==gfMemScore[i]){
                continue;
            }
            if( (0==stCycleFstSecThd.fData) && (gfMemScore[i]!=0) ){
                //stCycleFstSecThd.fData = gfMemScore[i];
                return i; // 返回第一个非0的导数值。
            }//[End] 3.1.1 

            //if(gfMemScore[i] > 1.2*stCycleFstSecThd.fData){
			//if(gfMemScore[i] > 1.1*stCycleFstSecThd.fData){ //@2019-01-13 解决起点比较平缓的情况
                //nOverChangeFirst = 1;
                //for(j=0; j<3; j++){
                //    f_local_sum += gfMemScore[i+j-1];                
                //}
                //if(f_local_sum/3 > 1.1*stCycleFstSecThd.fData){
                //if(f_local_sum/2 > 1.05*stCycleFstSecThd.fData){
                //    nOverChangeAvg = 1;
                //}
                //nOverChangeAvg = 1; //@2019-01-13 解决起点比较平缓的情况
            //}

            //if((nOverChangeAvg >= 1) && (gfMemScore[i-1] <= gfMemScore[i])){
                // When nOverChange==3, find the MAX-Point, otherwise nOverChange<3 .
                // and [i] <= [i+1]
                // If find the MAX-Point, return the Index of U0[i].
                //break;
                //return i;
                //@2019-01-23 Check the nIndex, avoid of early-find.
                //if(unit_detect_algorithm_correlator(i, gfMemData, UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM) < UNIT_DETECT_CORRELATOR_THRESHOLD ){
                //@2019-01-25 Renew, use gfMemScore instead of gfMemData.
                //if(unit_detect_algorithm_correlator(i, gfMemScore, UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM) < UNIT_DETECT_CORRELATOR_THRESHOLD ){
                //    return i;
                //}
            //}else if(nOverChangeFirst >= 1){
                // 仅更新待比较的 Score 值  @2019-01-13 不再更新 score，则仍用 1~3 周波内的最大 score
                //stCycleFstSecThd.fData = gfMemScore[i];
            //}

            //nOverChangeAvg = 0; // 清零，准备下一次的 j 循环。
            //nOverChangeFirst = 0;
        }
        // @2019-01-27 
        return unit_detect_algorithm_anomalies_range(gfMemScore , UNIT_DETECT_MAX_FAULTWAVE_FRAME_NUM);
        
    }else{
        return 0; //  Fetal error.
    }
	return 0; //  Fetal error.
}

/**
 * API Run the abnormal detect algorithm.
 * ---------- 
 * Return： the Index of SuddenChange-U0 : short
 */
short unit_detect_algorithm_run(float *p_data, short n_len){
    unsigned short nIndex=0, i=0;
    
    // Clear Container
    memset(gfMemScore, 0, sizeof(gfMemScore));
    memset(gfMemDerivatives, 0, sizeof(gfMemDerivatives));
    memset(gfMemData, 0, sizeof(gfMemData));

    // Recv Data & Store
    for(i=0; i<n_len; i++){
    	gfMemData[i] = *(p_data+i); 
	}
    
    // Calc Scores
    unit_detect_set_scores(gfMemScore);

    //# if not score_only: then { self._detect_anomalies() }
    nIndex = unit_detect_anomalies();
    //printf("Find the Index is %d \r\n", nIndex);
    
    return nIndex;
}

/**
 * Calc the correlator and Check the MAX-Abnormal-Point.
 * If correlator >= 0.5 means the current nIndex is not proper, and 
 * need find the next one.
 * ----------
 * Parameters :
 * nIndex : unsigned short
 *   the current nIndex of MAX-Abnormal-Point-Starter
 * *p_data : float
 *   U0-Score data.
 * n_len : short
 *   U0-Score data length.
 * ----------
 * Return ：correlator, is float type. [-1.0 ~ 1.0]
 */
static float unit_detect_algorithm_correlator(unsigned short nIndex, float *p_data, short n_len)
{
    float f_correlator = 0.0;
    unsigned short i=0, j=0;
    unsigned short n_local_num = 3; // Before and After the MAX-Abnormal-Point
    //float *p_data_before = (p_data + nIndex - UNIT_DETECT_PER_FAULTWAVE_FRAME_NUM - n_local_num);
    float *p_data_before = p_data;
    float *p_data_current = (p_data + nIndex - n_local_num);
    float sum_up = 0.0, sum_down = 0.0, sum_down_a = 0.0,  sum_down_b = 0.0;
    float average_a = 0.0, average_b = 0.0;

    for(j=0; j<UNIT_DETECT_PER_FAULTWAVE_FRAME_NUM*4; j++){
        sum_up = 0.0;
        sum_down = 0.0;
        sum_down_a = 0.0;
        sum_down_b = 0.0;
        average_a = 0.0;
        average_b = 0.0;
        // j1. 从数据Data的起始开始，遍历是否有相似度 >= 0.5 的情况； 这样是避免了考虑 周期 N，因为故障前波形的周期不一定完全=50Hz
        // j2. 仅需比对前 4 个周波数据即可
        for(i=0; i<(n_local_num*2+1); i++){
            average_a += *(p_data_before+j+i);
            average_b += *(p_data_current+i);
        }
        average_a /= (n_local_num*2+1);
        average_b /= (n_local_num*2+1);

        for(i=0; i<(n_local_num*2+1); i++){
            sum_up += (*(p_data_before+j+i) - average_a) * (*(p_data_current+i) - average_b);
            sum_down_a += (*(p_data_before+j+i) - average_a) * (*(p_data_before+j+i) - average_a);
            sum_down_b += (*(p_data_current+i) - average_b) * (*(p_data_current+i) - average_b);
        }
        
        sum_down = sum_down_a * sum_down_b;

        if(sum_down > 0){
            // 1. 
            sum_down = sqrtf(sum_down);
            f_correlator = sum_up/sum_down;
        }else if(sum_down_a != sum_down_b){
            // 2. means that only sum_down_a==0 or only sum_down_b==0
            f_correlator = 0.0;
        }else{
            // 3. means that ALL sum_down_a==0  and sum_down_b==0, this will not occur.
            f_correlator = 1.0;
        }

        //printf("f_correlator = %f \t nIndex = %d \t f_correlator = %f \r\n", f_correlator, nIndex, f_correlator);

        if(f_correlator >= UNIT_DETECT_CORRELATOR_THRESHOLD){
            break;
        }
    }

    //printf("f_correlator = %f \t nIndex = %d \r\n", f_correlator, nIndex);
    return f_correlator;
}

/**
 * Detect anomalies using a threshold on anomaly scores.
 * ----------
 * Parameters : 
 * *p_data : float
 *   U0-Score data, such as gfMemScore[]
 * n_len : short
 *   U0-Score data length.
 * ----------
 * Return ：unsigned short nIndex, the U0-Max-Starter in Cycle4~5.
 */
static unsigned short unit_detect_algorithm_anomalies_range(float *p_data, short n_len)
{
    float threshold = stCycleFstSecThd.fData; // == stCycleFstSecThd.pData of the MAX-Pointer data. Now assume !=0
    unsigned short i=0, j=0;
    unsigned short n_count = 0; // <=3, 
    unsigned short n_start = 0, n_end = 0;
    float f_sum = 0.0;
        
    // s1. # Find all the anomaly intervals.
    memset(stRangeIndex, 0, sizeof(stRangeIndex));
	// @2019-01-29 1.有些U0起点比较靠近第640点, 所以这里需要多判断 1/4 周波, 比如 {中国电科院/BAY00_1226_20180106_100858_981.cfg}
    for(i=0; i<UNIT_DETECT_PER_FAULTWAVE_FRAME_NUM*5+32; i++){
        if(p_data[i] > threshold){
            n_end = i;
            f_sum += p_data[i];
            if(0 == n_start){
                n_start = i;
            } // if(0 == n_start) ...
        }else if( (n_start != 0) && (n_end != 0) ){
            stRangeIndex[n_count].index_start = n_start;
            stRangeIndex[n_count].index_end = n_end;
            stRangeIndex[n_count].anomaly_score = f_sum;            
            n_start = n_end = 0;
            f_sum = 0.0;            
            n_count = n_count + 1;
            if(n_count > UNIT_DETECT_RANGE_INDEX_NUM){
                break; // Only Record UNIT_DETECT_RANGE_INDEX_NUM Array.
            }
        }// else if
    }

    // s2. Now have find 3 the anomaly intervals.
    unsigned short n_pos = 0; // Record the Position of MAX-Point.
    printf("unit_detect_algorithm_anomalies_range(...) \t threshold = %f \r\n", threshold);
    for(i=0; i<UNIT_DETECT_RANGE_INDEX_NUM; i++){
        printf("stRangeIndex[i].index_start = %d \t stRangeIndex[i].index_end = %d \t stRangeIndex[i].anomaly_score = %f \r\n", 
            stRangeIndex[i].index_start, stRangeIndex[i].index_end, stRangeIndex[i].anomaly_score );

        if(stRangeIndex[i].anomaly_score >= 1.5*threshold){
            for(j=stRangeIndex[i].index_start; j<=stRangeIndex[i].index_end; j++){
                if(p_data[j] >= 1.05*threshold){
                    n_pos = j; // Find the MAX-Point !!!
                    return n_pos;
                }
            }
        }
    }
    
    if(0 == n_pos){
        // s3. Above of all, there is no 1.5*threshold, so decrease to 1.05*threshold
        for(i=0; i<UNIT_DETECT_RANGE_INDEX_NUM; i++){           
            if(stRangeIndex[i].anomaly_score >= 1.05*threshold){
                for(j=stRangeIndex[i].index_start; j<=stRangeIndex[i].index_end; j++){
                    if(p_data[j] > threshold){
                        n_pos = j; // Find the MAX-Point !!!
                        return n_pos;
                    }
                }
            }
        }
    }

    //
    return n_pos;
}

/**
 * Test Case Func.
 * ----------
 * Output : Hello, world.
 */
void unit_unit_detect_test_case()
{
    printf("Hello, world. This is U0 C code. 2019-01-25 18:53 \r\n");
    return;
}

/**
 * 
 */
#if 0
void main(){
    // Test [Begin]
    float fTmpData[1664] = {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 4, 6, 8, 11, 15, 18, 22, 27, 32, 37, 40, 47, 55, 61, 67, 72, 80, 89, 96, 103, 110, 118, 127, 135, 142, 150, 158, 167, 174, 181, 191, 198, 206, 212, 220, 226, 234, 240, 246, 253, 258, 263, 268, 273, 278, 282, 285, 288, 291, 293, 295, 297, 298, 299, 298, 298, 298, 297, 295, 292, 291, 287, 284, 280, 275, 270, 265, 260, 253, 248, 241, 234, 225, 218, 211, 202, 192, 183, 176, 166, 156, 144, 135, 127, 117, 105, 94, 85, 75, 64, 51, 42, 34, 23, 11, 1, -6, -14, -26, -36, -44, -51, -59, -68, -77, -83, -89, -96, -103, -109, -114, -119, -124, -127, -132, -134, -137, -140, -141, -142, -143, -144, -144, -143, -142, -141, -138, -135, -135, -131, -126, -123, -117, -113, -107, -101, -96, -90, -82, -75, -68, -60, -53, -45, -36, -27, -19, -12, -3, 7, 16, 24, 31, 42, 51, 60, 68, 77, 86, 95, 103, 110, 118, 126, 134, 141, 148, 155, 161, 166, 172, 178, 183, 188, 191, 195, 199, 201, 204, 206, 208, 209, 210, 210, 210, 209, 207, 206, 204, 203, 198, 195, 192, 188, 183, 178, 173, 167, 162, 154, 146, 141, 134, 126, 116, 108, 101, 93, 82, 71, 63, 55, 46, 34, 24, 16, 7, -4, -15, -23, -31, -42, -52, -62, -70, -77, -88, -97, -105, -112, -119, -127, -135, -141, -147, -153, -158, -165, -169, -173, -177, -181, -184, -187, -189, -191, -192, -192, -193, -192, -192, -191, -189, -188, -185, -182, -179, -176, -171, -166, -162, -157, -150, -143, -136, -131, -126, -117, -107, -99, -93, -86, -75, -66, -57, -51, -41, -30, -20, -11, -4, 6, 16, 26, 34, 42, 53, 62, 70, 78, 86, 95, 103, 111, 117, 124, 131, 138, 143, 149, 154, 159, 164, 167, 171, 174, 177, 179, 181, 183, 183, 184, 184, 183, 182, 180, 179, 177, 174, 171, 167, 163, 159, 153, 148, 144, 137, 131, 122, 116, 110, 101, 92, 84, 78, 69, 58, 49, 40, 32, 22, 12, 2, -5, -14, -26, -36, -45, -52, -62, -73, -82, -90, -97, -107, -116, -124, -132, -138, -146, -153, -159, -165, -171, -176, -181, -186, -189, -194, -197, -200, -203, -204, -206, -207, -208, -208, -208, -207, -206, -204, -202, -199, -195, -193, -189, -185, -179, -175, -171, -164, -157, -151, -146, -139, -132, -123, -116, -110, -102, -92, -82, -75, -68, -59, -48, -38, -30, -23, -13, -2, 6, 13, 22, 32, 42, 50, 56, 66, 74, 82, 90, 97, 105, 112, 119, 124, 131, 136, 142, 147, 152, 156, 160, 164, 166, 169, 172, 174, 176, 177, 177, 176, 178, 176, 175, 173, 171, 170, 167, 164, 159, 157, 152, 147, 142, 136, 132, 126, 118, 111, 104, 98, 91, 82, 72, 65, 59, 49, 39, 30, 23, 15, 4, -5, -13, -21, -31, -42, -50, -58, -66, -76, -86, -93, -101, -109, -118, -125, -133, -139, -147, -153, -160, -165, -170, -176, -181, -186, -190, -194, -198, -200, -203, -204, -207, -208, -210, -210, -210, -210, -209, -208, -206, -204, -202, -198, -195, -192, -187, -182, -178, -174, -166, -159, -154, -149, -142, -134, -124, -118, -111, -102, -92, -83, -75, -67, -57, -46, -38, -30, -21, -9, 0, 9, 16, 27, 37, 46, 53, 61, 70, 80, 88, 95, 102, 110, 117, 123, 129, 136, 142, 147, 152, 157, 161, 164, 167, 170, 172, 175, 176, 177, 178, 179, 177, 177, 176, 174, 172, 170, 166, 164, 159, 154, 150, 146, 139, 133, 127, 122, 116, 107, 99, 92, 86, 77, 67, 58, 51, 44, 32, 22, 14, 6, -3, -14, -24, -31, -39, -51, -60, -69, -77, -85, -95, -103, -112, -120, -127, -135, -142, -149, -154, -161, -168, -173, -178, -182, -188, -192, -195, -197, -201, -203, -205, -207, -207, -208, -208, -208, -207, -206, -204, -202, -200, -196, -194, -188, -185, -180, -175, -169, -164, -158, -151, -144, -135, -129, -123, -114, -104, -96, -88, -81, -70, -59, -51, -44, -34, -23, -12, -4, 4, 14, 25, 33, 41, 50, 60, 69, 77, 84, 93, 101, 109, 116, 122, 129, 135, 142, 146, 152, 157, 161, 165, 168, 172, 174, 176, 178, 179, 180, 180, 180, 179, 179, 177, 174, 172, 170, 166, 161, 158, 153, 148, 142, 137, 132, 125, 117, 109, 104, 97, 89, 78, 70, 62, 55, 44, 34, 25, 18, 8, -2, -12, -20, -29, -40, -50, -58, -66, -76, -85, -93, -102, -109, -118, -127, -134, -141, -147, -154, -161, -166, -172, -177, -182, -186, -190, -193, -196, -199, -202, -203, -204, -206, -206, -205, -205, -204, -203, -200, -199, -196, -193, -188, -185, -180, -175, -169, -164, -159, -153, -144, -136, -131, -124, -115, -105, -97, -90, -82, -71, -61, -52, -45, -36, -24, -15, -5, 1, 13, 22, 32, 39, 48, 59, 68, 76, 85, 92, 101, 109, 115, 122, 128, 136, 142, 148, 152, 158, 162, 166, 170, 172, 176, 178, 180, 181, 182, 182, 182, 182, 181, 180, 178, 175, 172, 169, 165, 161, 158, 152, 146, 141, 135, 129, 121, 113, 107, 101, 92, 82, 75, 67, 59, 48, 38, 30, 22, 12, 0, -8, -16, -24, -36, -46, -55, -62, -72, -82, -90, -99, -106, -115, -124, -131, -137, -144, -151, -158, -164, -169, -174, -180, -184, -188, -191, -194, -197, -199, -201, -203, -203, -203, -203, -203, -202, -200, -198, -196, -194, -190, -186, -183, -178, -173, -167, -162, -157, -150, -142, -134, -128, -122, -112, -103, -95, -88, -80, -69, -59, -51, -43, -32, -22, -12, -2, 3, 15, 25, 35, 42, 52, 61, 71, 78, 86, 96, 103, 111, 119, 124, 132, 138, 145, 149, 155, 160, 165, 168, 172, 175, 178, 180, 182, 183, 184, 184, 185, 184, 183, 180, 180, 177, 174, 172, 167, 163, 159, 154, 147, 142, 137, 131, 123, 115, 109, 102, 94, 84, 75, 68, 60, 50, 39, 31, 23, 13, 2, -7, -15, -23, -35, -45, -53, -61, -70, -80, -90, -97, -105, -113, -122, -130, -137, -143, -150, -157, -163, -167, -172, -178, -182, -187, -189, -193, -195, -197, -199, -200, -201, -202, -202, -201, -200, -198, -196, -195, -191, -188, -183, -180, -176, -171, -165, -160, -154, -147, -140, -132, -126, -119, -110, -100, -92, -84, -76, -66, -56, -47, -39, -30, -18, -8, 0};
    short nIndex = 0; 
    nIndex = unit_detect_algorithm_run(fTmpData, 1664);
    printf("nIndex = %d", nIndex);

    system("PAUSE");
    return;
}
#endif //#if 0
