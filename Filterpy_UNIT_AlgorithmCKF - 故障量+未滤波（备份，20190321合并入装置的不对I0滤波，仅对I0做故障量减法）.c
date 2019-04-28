
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Constants var  &  Macro
#define UNIT_CKF_MAX_FAULTWAVE_FRAME_NUM 1664

/**
 * 计算突变量，后一个周波减去前一个周波；第一个周波的数据值完全赋0值。
*/
void unit_ckf_calc_sudden_change(float *pSampData, short nLenSamp)
{
    short i = 0;

    // 计算突变量，后一个周波减去前一个周波；第一个周波的数据值完全赋0值。
    for(i=nLenSamp-1; i>=128; i--){
        *(pSampData + i) = *(pSampData + i) - (*(pSampData + i - 128));
    }

    for(i; i>=0; i--){
        *(pSampData + i) = 0;
    }

}

/**
 * 对外 API ，用于调用 I0 Data。
 * ----------
 * Parameters
 * pSampData : short*
 *     [In] U0 或 I0 的原始采样点。
 * nLenSamp : short
 *     [In] U0 或 I0 的原始采样点的数据长度，数据的单位是 szie_t。
 * pSampDataNew : short*
 *     [Out] 滤波后的数据用于判别极性，并且是 float 强制转换为 short，为保留精度，所以扩大10倍。
 * bUsed : unsigned short 
 *     [In] bUsed 用于是否执行滤波的 flag, 若是为 true 则执行滤波; 若是为 false 则不执行滤波。
 */
void unit_ckf_process(short *pSampData , short nLenSamp, float *pSampDataNew, unsigned short bUsed)
{
    if((NULL == pSampData) || (nLenSamp <= 0) || (NULL == pSampDataNew)){
        return; //error
    }
     
    unsigned short i=0;
    unsigned short dim_x=2, dim_z=1, dt=2;
    
    for(i=0; i<nLenSamp; i++){    	
        *(pSampDataNew+i) = *(pSampData + i);
	}
    // Calc Sudden-Change, gap is 128    
    unit_ckf_calc_sudden_change(pSampDataNew, UNIT_CKF_MAX_FAULTWAVE_FRAME_NUM);

    // !!! Decide if use Filter...
    if(0 == bUsed){
        return;
    }
    
    return;	
}

