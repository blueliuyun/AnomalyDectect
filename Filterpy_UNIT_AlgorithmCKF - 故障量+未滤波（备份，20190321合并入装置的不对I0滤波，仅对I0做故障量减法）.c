
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Constants var  &  Macro
#define UNIT_CKF_MAX_FAULTWAVE_FRAME_NUM 1664

/**
 * ����ͻ��������һ���ܲ���ȥǰһ���ܲ�����һ���ܲ�������ֵ��ȫ��0ֵ��
*/
void unit_ckf_calc_sudden_change(float *pSampData, short nLenSamp)
{
    short i = 0;

    // ����ͻ��������һ���ܲ���ȥǰһ���ܲ�����һ���ܲ�������ֵ��ȫ��0ֵ��
    for(i=nLenSamp-1; i>=128; i--){
        *(pSampData + i) = *(pSampData + i) - (*(pSampData + i - 128));
    }

    for(i; i>=0; i--){
        *(pSampData + i) = 0;
    }

}

/**
 * ���� API �����ڵ��� I0 Data��
 * ----------
 * Parameters
 * pSampData : short*
 *     [In] U0 �� I0 ��ԭʼ�����㡣
 * nLenSamp : short
 *     [In] U0 �� I0 ��ԭʼ����������ݳ��ȣ����ݵĵ�λ�� szie_t��
 * pSampDataNew : short*
 *     [Out] �˲�������������б��ԣ������� float ǿ��ת��Ϊ short��Ϊ�������ȣ���������10����
 * bUsed : unsigned short 
 *     [In] bUsed �����Ƿ�ִ���˲��� flag, ����Ϊ true ��ִ���˲�; ����Ϊ false ��ִ���˲���
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

