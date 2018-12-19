
#include <math.h>
#include <stdlib.h>
#include <string.h>

//#include "Filterpy_UNIT_AlgorithmCKF.h"
//#include "Globevar.h"

// Private Var
/**
 * @brief Instance structure for the floating-point matrix structure.
 */
typedef struct
{
    unsigned short numRows;     /**< number of rows of the matrix.     */
    unsigned short numCols;     /**< number of columns of the matrix.  */
    float *pData;     /**< points to the data of the matrix. */
} MATRIX_F32_STRUCT;

// 1个 static 类型的结构体
typedef struct{
    MATRIX_F32_STRUCT smQ;
    MATRIX_F32_STRUCT smR; // Matrix
    MATRIX_F32_STRUCT saX; // Array
    MATRIX_F32_STRUCT smP;
    //self.x = zeros(dim_x)
    //self.P = eye(dim_x)
    unsigned short K;
    unsigned short dim_x;
    unsigned short dim_z;
    unsigned short dt;
    unsigned short _num_sigmas;
    MATRIX_F32_STRUCT smSigma;
    MATRIX_F32_STRUCT smS;
    MATRIX_F32_STRUCT smSI;
    //self.hx = hx
    //self.fx = fx
    //self.x_mean = x_mean_fn
    //self.z_mean = z_mean_fn
    unsigned short y;
    //self.z = np.array([[None]*self.dim_z]).T
    //self.S = np.zeros((dim_z, dim_z)) # system uncertainty
    //self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty
    MATRIX_F32_STRUCT smZ;
    MATRIX_F32_STRUCT sm_sigmas_f;
    MATRIX_F32_STRUCT sm_sigmas_h;
    MATRIX_F32_STRUCT saX_prior; // Array
    MATRIX_F32_STRUCT saX_post;  // Array
    MATRIX_F32_STRUCT smP_prior;
    MATRIX_F32_STRUCT smP_post;
}CKF_STRUCT;
static CKF_STRUCT sCKF;

//////////////////////////////////////////////////////////////
static void unit_mat_diagonal_f32(MATRIX_F32_STRUCT* s);
static void unit_mat_zero_f32(MATRIX_F32_STRUCT* s);
static /*float*/void eye(MATRIX_F32_STRUCT* s, unsigned short nCol, unsigned short nRow);
static unsigned short GetIndexInMatrix(MATRIX_F32_STRUCT* sm, unsigned short row, unsigned short col);
static void unit_ckf_transform(MATRIX_F32_STRUCT* sm_sigmas_f, \
    MATRIX_F32_STRUCT* sm_Q, MATRIX_F32_STRUCT* sa_x/*Array*/, MATRIX_F32_STRUCT* sm_P);

/**
 * Init
 * ----------
 * Return
 * ----------
 * Parameters
 * nDim_x : unsigned short
 *  Number of ...
 * nDim_z : unsigned short
 *  Number of of measurement inputs. For example, if the sensor
 *  provides you with position in (x,y), nDim_z would be 2.
 */
static void CubatureKalmanFilter(unsigned short nDim_x, unsigned short nDim_z, unsigned short nDt)
{
    // 每次使用 Init 时都需清空 static 类型的结构体; @2018-12-15 需要有选择的清0
    //memset(&sCKF, 0, sizeof(sCKF));

    eye(&sCKF.smQ, nDim_x, nDim_x);
    eye(&sCKF.smR, nDim_z, nDim_z);
    eye(&sCKF.saX, nDim_x, 1);
    eye(&sCKF.smP, nDim_x, nDim_x); // Now @2018-12-17 matrix==2x2
    sCKF.K = 0;
    sCKF.dim_x = nDim_x;
    sCKF.dim_z = nDim_z;
    sCKF.dt = nDt;
    sCKF._num_sigmas = 2*sCKF.dim_x;
    
    eye(&sCKF.smSigma, nDim_x*2, nDim_x);
    //self.hx = hx
    //self.fx = fx
    //self.x_mean = x_mean_fn
    //self.z_mean = z_mean_fn
    sCKF.y = 0;
    //self.z = np.array([[None]*self.dim_z]).T
    eye(&sCKF.smZ, nDim_z, 1);
    eye(&sCKF.smS, nDim_z, nDim_z);  // system uncertainty
    eye(&sCKF.smSI, nDim_z, nDim_z); // inverse system uncertainty

    // sigma points transformed through f(x) and h(x)
    // variables for efficiency so we don't recreate every update
    eye(&sCKF.sm_sigmas_f, nDim_x*2, nDim_x); // when nDim_x=2, then sm_sigmas_f is 4x2
    eye(&sCKF.sm_sigmas_h, nDim_x*2, nDim_z); // and nDim_z=1, then sm_sigmas_h is 4x1, see hx()

    // these will always be a copy of x,P after predict() is called
    eye(&sCKF.saX_prior, nDim_x, 1);
    eye(&sCKF.smP_prior, nDim_x, nDim_x);
    //sCKF.saX_prior = sCKF.saX;
    //sCKF.smP_prior = sCKF.smP;

    //  these will always be a copy of x,P after update() is called
    eye(&sCKF.saX_post, nDim_x, 1);
    eye(&sCKF.smP_post, nDim_x, nDim_x);
    //sCKF.saX_post = sCKF.saX;
    //sCKF.smP_post = sCKF.smP;
}

/**
 * def eye(N, M=None, k=0, dtype=float, order='C'):
 * ----------
 * Return a 2-D array with ones on the diagonal and zeros elsewhere.
 * ----------
 * Parameters
 * nRow : unsigned short
 *   Number of rows in the output.
 * 
 */
static void eye(MATRIX_F32_STRUCT* s, unsigned short nRow, unsigned short nCol)
{
    // Init MATRIX_F32_STRUCT
    // 1. Check s != NULL
    s->numCols = nCol;
    s->numRows = nRow;
    
    unit_mat_zero_f32(s); // All is 0
    //unit_mat_diagonal_f32(s); // 对角线是 1， 其他是 0 值
}

/**
 * @2018-12-15 Later, Need to check MEMORY.
 */ 
static void unit_mat_zero_f32(MATRIX_F32_STRUCT* s)
{
	unsigned short pos = 0;
	unsigned short blockSize = s->numRows * s->numCols;
	float *A = (float*)malloc(blockSize * sizeof(float));

	do{
		A[pos] = 0.0f;
		pos++;
	} while (pos < blockSize);

	s->pData = A;
}

/**
 * @2018-12-15 Later, Need to check MEMORY.
 * array([[1, 0],
 *        [0, 1]])
 */ 
static void unit_mat_diagonal_f32(MATRIX_F32_STRUCT* s)
{
	unsigned short pos = 0, rowSeq = 0;
	unsigned short blockSize = s->numRows * s->numCols;
	float *A = s->pData;

	do{
		A[pos] = 1.0f;
        rowSeq++;
		pos = s->numCols * rowSeq + rowSeq;
	} while (pos < blockSize);
}

/**
 * Compute the Cholesky decomposition of a matrix.
 * ----------
 * 返回值。
 */
static void cholesky(MATRIX_F32_STRUCT* smP, MATRIX_F32_STRUCT* smU)
{

}

/**
 * Common code for cholesky() and cho_factor().
 * ----------
 * 返回值。
 */
static void _cholesky(MATRIX_F32_STRUCT* smP, MATRIX_F32_STRUCT* smU)
{    
    // smp == a matrix 2x2, 
    if(smP->numRows != 2){
        return; // Need one error staus; Matrix must be square.
    }

    // Squareness check
    if(smP->numRows != smP->numCols){
        return; // Need one error staus
    }

    unsigned short nRows = smP->numRows;
    unsigned short i = 0;
    float a[4] = { 0.0f }; // smp == a matrix 2x2
    memcpy(a, smP->pData, 4*sizeof(float));

    for(i = 0; i < nRows; i++) {
        unsigned short ii = GetIndexInMatrix(smP, i, i);
        unsigned short k = 0;
        for (/*unsigned short*/ k = 0; k < i; k++) {
            unsigned short ki = GetIndexInMatrix(smP, k, i);
            a[ii] = a[ii] - a[ki] * a[ki];
        }

        if (a[ii] < 0) {
            //error , throw std::runtime_error("Matrix is not positive definite.");
            return;
        }

        a[ii] = sqrt(a[ii]);
        unsigned short j=0;
        for (/*unsigned short*/ j = i + 1; j < nRows; j++) {
            unsigned short ij = GetIndexInMatrix(smP, i, j);
            unsigned short k=0;
            for (/*unsigned short*/ k = 0; k < i; k++) {
                unsigned short ki = GetIndexInMatrix(smP, k, i);
                unsigned short kj = GetIndexInMatrix(smP, k, j);
                a[ij] = a[ij] - a[ki] * a[kj];
            }
            if (a[ij] != 0) a[ij] = a[ij] / a[ii];
        }
    }
    
    // Clear out the lower matrix
    unsigned short j=0;
    for (i = 1; i < nRows; i++) {
        for (j = 0; j < i; j++) {
            unsigned short ij = GetIndexInMatrix(smP, i, j);
            a[ij] = 0;
        }
    }
    
    memcpy(smU->pData, a, 4*sizeof(float));
    return ; //Matrix(n, n, a);
}

/**
 * ----------
 * 返回值是 [row][col] 在数组内存上的 Index。
 * ----------
 * Parameters
 * sm : MATRIX_F32_STRUCT*
 *  the pointer of matrix.
 */
static unsigned short GetIndexInMatrix(MATRIX_F32_STRUCT* sm, unsigned short row, unsigned short col)
{
  if (sm) return row * sm->numCols + col;
  else return col * sm->numRows + row;
}

/**
 * Creates cubature points for the the specified state and covariance.
 * ----------
 * 返回值类型是数组。
 */ 
static void unit_ckf_spherical_radial_sigmas(MATRIX_F32_STRUCT* saX, MATRIX_F32_STRUCT* smP, MATRIX_F32_STRUCT* smSigma)
{
    // get Rows of P
    unsigned short nRows = smP->numRows;  // n, _ = P.shape
    unsigned short nCols = smP->numCols;
    unsigned short i=0, j=0; 
    float *pData = saX->pData;

    MATRIX_F32_STRUCT local_smU; // local, MATRIX_smU, 2x2
    eye(&local_smU, nRows, nRows);
    //eye(smSigma, nRows*2, nRows); // MATRIX_smSigma, 4x2
    //@2018-12-16 若 P 是单个元素的矩阵, 则矩阵 P 一定是正定矩阵。
    //U = cholesky(smP) * sqrt(n)
    _cholesky(smP, &local_smU); //@2018-12-18 _cholesky has been Checked √
    for(i=0; i<(local_smU.numRows*local_smU.numCols); i++){ // Now Total size == 2x2
        local_smU.pData[i] = local_smU.pData[i] * sqrt(nRows);
    }
    for(i=0; i<nRows; i++){
        for(j=0; j<nCols; j++){
            // 矩阵元素依次对应相加
            *(smSigma->pData+(i*nRows+j)) = *(pData+j)  + *(local_smU.pData+(i*nCols+j));
            *(smSigma->pData+((i+nRows)*nCols+j)) = *(pData+j) - *(local_smU.pData+(i*nCols+j));
        }
    }
}

/**
 * ----------
 */
static void unit_ckf_fx(float* pData_f, float* pData)
{
    // Received & Deal the Row Items   
    *(pData_f) = *(pData+1) * sCKF.dt + *(pData);
    *(pData_f+1) = *(pData+1);
}

/**
 * Performs the predict step of the CKF. On return, self.x and
 * self.P contain the predicted state (x) and covariance (P).
 * ----------
 * Important: this MUST be called before update() is called for the first time.
 */
static void unit_ckf_predict()
{
    unit_ckf_spherical_radial_sigmas(&sCKF.saX, &sCKF.smP, &sCKF.smSigma);
    
    // evaluate cubature points
    unsigned short k=0;
    for(k=0; k<sCKF._num_sigmas; k++){
        unit_ckf_fx(sCKF.sm_sigmas_f.pData+(sCKF.sm_sigmas_f.numCols*k), sCKF.smSigma.pData+(sCKF.smSigma.numCols*k));
    }

    //self.x, self.P = ckf_transform(self.sigmas_f, self.Q)
    unit_ckf_transform(&sCKF.sm_sigmas_f, &sCKF.smQ, &sCKF.saX, &sCKF.smP);

    // save prior
    memcpy(sCKF.saX_prior.pData, sCKF.saX.pData, sCKF.saX.numRows*sCKF.saX.numCols*sizeof(float));
    memcpy(sCKF.smP_prior.pData, sCKF.smP.pData, sCKF.smP.numRows*sCKF.smP.numCols*sizeof(float));
}

/**
 * Compute Matrix outer.
 * ----------
 * 1. 重要：求内积的操作之前，MATRIX_F32_STRUCT* sm 要已经被初始化并分配了内存空间。
 * 2. MATRIX_F32_STRUCT* sm 的是 MxM 维矩阵，其中 M = Row*Col
 */
static void unit_ckf_matrix_outer(float* pData, 
    unsigned short size, MATRIX_F32_STRUCT* sm)
{
    unsigned short i=0, j=0;
    
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            *(sm->pData+i*size+j) = *(pData+j) * (*(pData+i));
        }
    }   
}

/**
 * Compute mean and covariance of array of cubature points.
 * ----------
 */
static void unit_ckf_transform(MATRIX_F32_STRUCT* sm_sigmas_f, \
    MATRIX_F32_STRUCT* sm_Q, MATRIX_F32_STRUCT* sa_x/*Array*/, MATRIX_F32_STRUCT* sm_P)
{
    unsigned short i=0, j=0;
    unsigned short mRow=0, nCol=0;
    mRow = sm_sigmas_f->numRows;
    nCol = sm_sigmas_f->numCols;
       
    // s1===> Xs : 4x2     s2===> x = sum(Xs, 0)[:, None] / m     s3===> x : 2x1
    for(i=0; i<nCol; i++){
        for(j=0; j<mRow; j++){
            *(sa_x->pData+i) += *(sm_sigmas_f->pData + j*nCol + i);    
        }
        *(sa_x->pData+i) /= mRow;
    }

    //P = np.zeros((n, n))
    //xf = x.flatten(); // 2x1 ===> 1,1
    float *pxf = sa_x->pData;
    // @2018-12-18 临时存储矩阵内存计算的结果， 需要在函数结尾处 free 内存。
    MATRIX_F32_STRUCT smTmpXs, smTmpxf;
    unsigned short size = sa_x->numRows*sa_x->numCols;
    eye(&smTmpXs, size, size);
    eye(&smTmpxf, size, size);
    // 需要全部清零
    memset(sm_P->pData, 0, sm_P->numRows*sm_P->numCols);
    for(i=0; i<mRow; i++){
        // 内积
        unit_ckf_matrix_outer((sm_sigmas_f->pData+ i*sm_sigmas_f->numCols), size, &smTmpXs);
        unit_ckf_matrix_outer((sa_x->pData/*+ i*sa_x->numCols*/), size, &smTmpxf);
        // P + 矩阵加
        for(j=0; j<(sm_P->numRows*sm_P->numCols); j++){
            *(sm_P->pData+j) += *(smTmpXs.pData+j) - (*(smTmpxf.pData+j));
        }
    }

    for(j=0; j<(sm_P->numRows*sm_P->numCols); j++){
        *(sm_P->pData+j) /= mRow;
        *(sm_P->pData+j) += *(sm_Q->pData+j); // P、Q 行列个数相等
    }
    // Need free { MATRIX_F32_STRUCT smTmpXs, smTmpxf; } Memory.
}

/**
 * Get the first Item of every Row.
 * ----------
 */
static void unit_ckf_hx(MATRIX_F32_STRUCT* sm)
{

}

/**
 * Inverse Square-MATRIX. Compute the Inverse Square-MATRIX of smSrc, and save smDest.
 * ----------
 * Parameters
 * smSrc : MATRIX_F32_STRUCT*
 *  the source of matrix pointer.
 * smDest : MATRIX_F32_STRUCT*
 *  the destination of matrix pointer.
 */
static void unit_ckf_inverse(MATRIX_F32_STRUCT* smSrc, MATRIX_F32_STRUCT* smDest)
{

}

/**
 * Update the CKF with the given measurements. On return, self.x
 * and self.P contain the new mean and covariance of the filter.
 * ----------
 * Important: predict() MUST be called before update() is called for the first time.
 */
static void unit_ckf_update()
{
    unsigned short i=0;
    //sCKF.saX
    for(i=0; i<sCKF._num_sigmas; i++){
        // self.sigmas_h[k] = self.hx(self.sigmas_f[k], *hx_args)
        // Now sm_sigmas_h is 4x1, 
        // Get the first Item of every Row. == unit_ckf_hx()
        *(sCKF.sm_sigmas_h.pData+i) = *(sCKF.sm_sigmas_f.pData+i*sCKF.sm_sigmas_f.numCols);
    }

    //# mean and covariance of prediction passed through unscented transform.
    //zp, self.S = ckf_transform(self.sigmas_h, R)
    MATRIX_F32_STRUCT zp; // Local var, and Malloc Memory, then later must be Free Memroy manully.
    eye(&zp, sCKF.sm_sigmas_h.numRows, 1); // sm_sigmas_h : 4x1,
    unit_ckf_transform(&sCKF.sm_sigmas_h, &sCKF.smR, &zp, &sCKF.smS);
    //# self.SI = inv(self.S)
    unit_ckf_inverse(&sCKF.smS, &sCKF.smSI); // Get the Inverse.
    

    //# compute cross variance of the state and the measurements.
    
}

/**
 * Test Func.
 * ----------
 */
#if 1
void main(){
    MATRIX_F32_STRUCT local_smU;

    CubatureKalmanFilter(2, 1, 2);

#if 0
	eye(&local_smU, 2, 2);
		
	*(sCKF.smP.pData) = 1;
	*(sCKF.smP.pData+1) = -2;
	*(sCKF.smP.pData+2) = 2;
	*(sCKF.smP.pData+3) = 5;
	_cholesky(&sCKF.smP, &local_smU);
	
    unit_ckf_predict();

	for(unsigned short i=0; i<(local_smU.numRows*local_smU.numCols); i++){ // Now Total size == 2x2
        local_smU.pData[i] = local_smU.pData[i] * sqrt(2);
    }
    
    *(sCKF.saX.pData) = 3;
    *(sCKF.saX.pData+1) = 5;
    
    for(unsigned short i=0; i<2; i++){
        for(unsigned short j=0; j<2; j++){
            *(sCKF.smSigma.pData+(i*2+j)) = *(sCKF.saX.pData+j)  + *(local_smU.pData+(i*2+j));
            *(sCKF.smSigma.pData+((i+2)*2+j)) = *(sCKF.saX.pData+j) - *(local_smU.pData+(i*2+j));
        }
    }
#endif // #if 0

    // Test 
    MATRIX_F32_STRUCT smTmpXs;
    unsigned short size = sCKF.saX.numCols*sCKF.saX.numRows;
    eye(&smTmpXs, size, size);
    unit_ckf_matrix_outer(sCKF.saX.pData, size, &smTmpXs);

    // Test
    for(unsigned short i=0; i<sCKF.sm_sigmas_f.numRows*sCKF.sm_sigmas_f.numCols; i++){
       *(sCKF.sm_sigmas_f.pData+i) = i;
    }
    unit_ckf_update();
} 
#endif

