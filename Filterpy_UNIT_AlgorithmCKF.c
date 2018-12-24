
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    //self.P = eyeZero(dim_x)
    MATRIX_F32_STRUCT smK; // Matrix
    MATRIX_F32_STRUCT smKT; // Matrix.K.T （K的转置）
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
    MATRIX_F32_STRUCT smY;
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
static void eyeZero(MATRIX_F32_STRUCT* s, unsigned short nRow, unsigned short nCol);
static void eyeE(MATRIX_F32_STRUCT* s, unsigned short nRow, unsigned short nCol);
static unsigned short GetIndexInMatrix(MATRIX_F32_STRUCT* sm, unsigned short row, unsigned short col);
static void unit_ckf_transform(MATRIX_F32_STRUCT* sm_sigmas_f, \
    MATRIX_F32_STRUCT* sm_Q, MATRIX_F32_STRUCT* sa_x/*Array*/, MATRIX_F32_STRUCT* sm_P);
static void unit_ckf_predict();

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
static void unit_ckf_CubatureKalmanFilter(unsigned short nDim_x, unsigned short nDim_z, unsigned short nDt)
{
    // 每次使用 Init 时都需清空 static 类型的结构体; @2018-12-15 需要有选择的清0
    //memset(&sCKF, 0, sizeof(sCKF));

    eyeE(&sCKF.smQ, nDim_x, nDim_x);
    eyeE(&sCKF.smR, nDim_z, nDim_z);
    eyeZero(&sCKF.saX, nDim_x, 1);
    eyeE(&sCKF.smP, nDim_x, nDim_x); // Now @2018-12-17 matrix==2x2
    eyeZero(&sCKF.smK, nDim_x, nDim_z); // Now @2018-12-21 matrix==2x1
    eyeZero(&sCKF.smKT, nDim_z, nDim_x); // Now @2018-12-21 matrix.T==1x2
    sCKF.dim_x = nDim_x;
    sCKF.dim_z = nDim_z;
    sCKF.dt = nDt;
    sCKF._num_sigmas = 2*sCKF.dim_x;
    
    eyeZero(&sCKF.smSigma, nDim_x*2, nDim_x);
    //self.hx = hx
    //self.fx = fx
    //self.x_mean = x_mean_fn
    //self.z_mean = z_mean_fn    
    eyeZero(&sCKF.smY, 1, 1);    
    //self.z = np.array([[None]*self.dim_z]).T
    eyeZero(&sCKF.smZ, nDim_z, 1);
    eyeZero(&sCKF.smS, nDim_z, nDim_z);  // system uncertainty
    eyeZero(&sCKF.smSI, nDim_z, nDim_z); // inverse system uncertainty

    // sigma points transformed through f(x) and h(x)
    // variables for efficiency so we don't recreate every update
    eyeZero(&sCKF.sm_sigmas_f, nDim_x*2, nDim_x); // when nDim_x=2, then sm_sigmas_f is 4x2
    eyeZero(&sCKF.sm_sigmas_h, nDim_x*2, nDim_z); // and nDim_z=1, then sm_sigmas_h is 4x1, see hx()

    // these will always be a copy of x,P after predict() is called
    eyeZero(&sCKF.saX_prior, nDim_x, 1);
    eyeZero(&sCKF.smP_prior, nDim_x, nDim_x);
    //sCKF.saX_prior = sCKF.saX;
    //sCKF.smP_prior = sCKF.smP;

    //  these will always be a copy of x,P after update() is called
    eyeZero(&sCKF.saX_post, nDim_x, 1);
    eyeZero(&sCKF.smP_post, nDim_x, nDim_x);
    //sCKF.saX_post = sCKF.saX;
    //sCKF.smP_post = sCKF.smP;
}

/**
 * def eyeZero(N, M=None, k=0, dtype=float, order='C'):
 * ----------
 * Return a 2-D array with ones on the diagonal and zeros elsewhere.
 * ----------
 * Parameters
 * nRow : unsigned short
 *   Number of rows in the output.
 */
static void eyeZero(MATRIX_F32_STRUCT* s, unsigned short nRow, unsigned short nCol)
{
    // Init MATRIX_F32_STRUCT
    // 1. Check s != NULL
    s->numCols = nCol;
    s->numRows = nRow;    
    unit_mat_zero_f32(s); // All is 0
}

/**
 * def eyeE(N, M=None, k=0, dtype=float, order='C'):
 * 生成一个单位矩阵 E，对角线元素全是 1 。
 * ----------
 * Return a 2-D array with ones on the diagonal and zeros elsewhere.
 * ----------
 * Parameters
 * nRow : unsigned short
 *   Number of rows in the output.
 * 
 */
static void eyeE(MATRIX_F32_STRUCT* s, unsigned short nRow, unsigned short nCol)
{
    // Init MATRIX_F32_STRUCT
    // 1. Check s != NULL
    s->numCols = nCol;
    s->numRows = nRow;    
    unit_mat_zero_f32(s); // All is 0
    unit_mat_diagonal_f32(s); // 对角线是 1， 其他是 0 值
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
 * ----------
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
 * The matrix_smDest is matrix_smSrc.T (transpose)
 * ----------
 * @2018-12-15 Later, Need to check MEMORY.
 */ 
static void unit_mat_transpose_A_B(MATRIX_F32_STRUCT* smDest, MATRIX_F32_STRUCT* smSrc)
{
    unsigned short i=0, j=0;
    float *pTmpData = smDest->pData;    
    for(i=0; i<smSrc->numCols; i++){
        for(j=0; j<(smSrc->numRows*smSrc->numCols); j+=smSrc->numCols){
            *(pTmpData++) = *(smSrc->pData + i + j);
        }
    }
    // Clear
    pTmpData = NULL;
}
/**
 * Compute the (matrixA + matrixB), So all is MxN.
 * ----------
 * @2018-12-15 Later, Need to check MEMORY.
 */ 
static void unit_mat_add_A_B(MATRIX_F32_STRUCT* smDest, \
    MATRIX_F32_STRUCT* smA, MATRIX_F32_STRUCT* smB)
{
	unsigned short i=0, j=0;
    for(i=0; i<smDest->numRows; i++){
        for(j=0; j<smDest->numCols; j++){
            *(smDest->pData + i*smDest->numCols + j) = *(smA->pData + i*smDest->numCols + j) + *(smB->pData + i*smDest->numCols + j);
        }
    }
}

/**
 * Compute the np.dot(matrixA, matrixB)
 * 矩阵运算 dot(M_A, M_B) 可能!=  M_A*M_B，应区别处理。
 * ----------
 * Parameters
 * smDest : MATRIX_F32_STRUCT*
 *  the pointer of matrix, which is used for Recv Resutl.
 */
static void unit_mat_dot_A_B(MATRIX_F32_STRUCT* smDest, \
    MATRIX_F32_STRUCT* smSrcA, MATRIX_F32_STRUCT* smSrcB)
{
    unsigned short i=0, j=0, k=0;

    memset(smDest->pData, 0, smDest->numRows*smDest->numCols*sizeof(float)); // Because of the following += OP, so need Clear Dest matrix.
    for(i=0; i<smSrcA->numRows; i++){
        for(j=0; j<smSrcB->numCols; j++){
            for(k=0; k<smSrcA->numCols; k++){
                *(smDest->pData + i*(smSrcB->numCols) + j) += *(smSrcA->pData + i*(smSrcA->numCols) + k) \
                    * (*(smSrcB->pData + k*(smSrcB->numCols) + j));
            }
        }
    }
}

/**
 * Compute the (matrixA)*(matrixB)
 * 矩阵运算 dot(M_A, M_B) 可能!=  M_A*M_B，应区别处理。
 * In unit_mat_AxB() 矩阵 A、B 行列应相等，且仅是对应元素相乘。
 * ----------
 * Parameters
 * smDest : MATRIX_F32_STRUCT*
 *  the pointer of matrix, which is used for Recv Resutl.
 */
static void unit_mat_AxB(MATRIX_F32_STRUCT* smDest, \
    MATRIX_F32_STRUCT* smSrcA, MATRIX_F32_STRUCT* smSrcB)
{
    unsigned short i=0, k=0;
    for(i=0; i<smSrcA->numRows; i++){        
        for(k=0; k<smSrcA->numCols; k++){
            *(smDest->pData + i*(smSrcA->numCols) + k) = *(smSrcA->pData + i*(smSrcA->numCols) + k) \
                * (*(smSrcB->pData + i*(smSrcA->numCols) + k));
        }
    }
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
    eyeZero(&local_smU, nRows, nRows);
    //eyeZero(smSigma, nRows*2, nRows); // MATRIX_smSigma, 4x2
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
 * Parameters
 * sm_sigmas_f : MATRIX_F32_STRUCT*
 *     [In] the pointer of matrix.
 * sm_Q : MATRIX_F32_STRUCT*
 *     [In]
 * sa_x : MATRIX_F32_STRUCT*
 *     [Out]
 * sm_P : MATRIX_F32_STRUCT*
 *     [Out]
 */
static void unit_ckf_transform(MATRIX_F32_STRUCT* sm_sigmas_f, \
    MATRIX_F32_STRUCT* sm_Q, MATRIX_F32_STRUCT* sa_x/*Array*/, MATRIX_F32_STRUCT* sm_P)
{
    unsigned short i=0, j=0;
    unsigned short mRow=0, nCol=0;
    mRow = sm_sigmas_f->numRows;
    nCol = sm_sigmas_f->numCols;
       
    // s1===> Xs : 4x2     s2===> x = sum(Xs, 0)[:, None] / m     s3===> x : 2x1
    // so, if s1===> sm_sigmas_f_h : 4x2_1     s2===> x = sum(Xs, 0)[:, None] / m     s3===> x : 2x1__1x1
    // thus, x.rows is == Xs.cols, and x.cols == 1.
    memset(sa_x->pData, 0, sa_x->numRows*sa_x->numCols*sizeof(float));
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
    eyeZero(&smTmpXs, size, size);
    eyeZero(&smTmpxf, size, size);
    // 需要全部清零
    memset(sm_P->pData, 0, sm_P->numRows*sm_P->numCols*sizeof(float));
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
 * Calc |Matrix| . Used for Inverse Square-MATRIX.
 * ----------
 * 1. When Call this func , must check the return value |Matrix| != 0
 * 2. In this func Rows==Cols .
 * ----------
 * Parameters
 * size : unsigned short
 *  In this func Rows==Cols==size .
 */
static float unit_ckf_calc_matrix_rank(float* pData, unsigned short size)
{
    unsigned short i=0, j=0, k=0;
    if(size==1) {
        return *(pData);
    }

    float retValue = 0.0;
    float* pLocalData = (float*)malloc(size*size*sizeof(float));
    memset(pLocalData, 0.0, (size*size*sizeof(float)));
    
    for(i=0; i<size; i++){
        for(j=0; j<size-1; j++){
            for(k=0; k<size-1; k++){
                *(pLocalData+ j*(size-1)+ k) = *( pData+ (j+1)*(size) + ((k>=i)?k+1:k));                
            }
        }
        float tmpValue = unit_ckf_calc_matrix_rank((float*)pLocalData, size-1);
        if(i%2 == 0){
            retValue += *(pData+i) * tmpValue;
        } else {
            retValue -= *(pData+i) * tmpValue;
        }
    }

    free(pLocalData); // Deallocate memory
    return retValue;
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
    if((smSrc->numRows!=smSrc->numCols) || (smDest->numRows!=smDest->numCols)){
        return; // error, Matrix must be square.
    }
    // Check |Matrix| != 0
    float fRank = 0.0;
    fRank = unit_ckf_calc_matrix_rank((float*)smSrc->pData, smSrc->numRows);
    if(0 == fRank){
        return; // error, fRank must not be 0.
    }

    unsigned short i=0, j=0, k=0, t=0;
    unsigned short size = smDest->numRows;
    if(1 == size){
        *(smDest->pData) = 1/fRank;
        return;
    }
    float* pLocalData = (float*)malloc(size*size*sizeof(float));
    memset(pLocalData, 0.0, (size*size*sizeof(float)));
    // 计算每一行每一列的每个元素所对应的余子式，组成 A* 
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            for(k=0; k<(size-1); k++){
                for(t=0; t<(size-1); t++){
                    *(pLocalData + k*(size-1)+ t) = *(smSrc->pData + (k>=i?k+1:k)*size + (t>=j?t+1:t));                    
                }
            }
            *(smDest->pData+ j*size+ i) = unit_ckf_calc_matrix_rank((float*)pLocalData, size-1) / fRank;
            if((i+j)%2 == 1){
                *(smDest->pData+ j*size+ i) = - *(smDest->pData+ j*size+ i);
            }
        }
    }
    // Must release Memroy.
    free(pLocalData);
}

/**
 * Computes the sum of the outer products of the rows in A and B
 * ----------
 * Parameters
 * pDataC : float*
 *  The Result Value(MxN) Address.
 * &rowC : unsigned short
 *  [Out] The Result Value M, rowC== colA
 * &colC : unsigned short
 *  [Out] The Result Value N, colC== colB
 */
static void unit_ckf_outer_product_sum(float* pDataA, unsigned short rowA, unsigned short colA, \
    float* pDataB, unsigned short rowB, unsigned short colB, \
    float* pDataC)
{
    if(pDataA == NULL || pDataB == NULL || pDataC == NULL){
        return; // error
    }

    // Used for store Outer-Matrix Processing Data
    float* pLocalData = (float*)malloc(rowA*colA*colB*sizeof(float));
    float* pTmpData = pLocalData; // Used for Move pointer, Now is Header.
    memset(pLocalData, 0.0, (rowA*colA*colB*sizeof(float)));
    
    //... ...
    unsigned short i=0, j=0, k=0, h=0;
    
    for(j=0; j<rowB; j++){ // 矩阵 rowA == rowB
        for(h=0; h<colA; h++){
            float dataA = *(pDataA + j*colA + h);                
            for(k=0; k<colB; k++){
                *(pTmpData++) = *(pDataB+ j*colB + k) * dataA;
            }
        }
    }
    

    pTmpData = pLocalData; // Used for Move pointer, Now is Header.
    for(i=0; i<colA; i++){
        for(j=0; j<colB; j++){            
            for(k=0; k<rowA; k++){
                *(pDataC+ i*colB + j) += *(pTmpData + i + k*colA);
			}
        }        
    }
    
    // Clear
    pTmpData = NULL;
    free(pLocalData);
}

/**
 * Used for matrix subtraction. 
 * matrix_A = matrix_B - matrix_C
 * ----------
 * Result : renew the { float* pDataA }
 */
static void unit_ckf_matrix_row_subtraction(float* pDataA, unsigned short rowA, unsigned short colA, \
    float* pDataB, unsigned short rowB, unsigned short colB, \
    float* pDataC, unsigned short rowC, unsigned short colC)
{
    unsigned short i=0, j=0;

    if(rowB != rowC){
        // 1. 矩阵B 与 矩阵C 的 rows 不相等时： 矩阵B的每行分别减去矩阵C（行向量减）
        for(i=0; i<rowB; i++){
            for(j=0; j<colB; j++){
                *(pDataA + i*colB + j) = *(pDataB + i*colB + j) - *(pDataC + j);
            }
        }
    } else {
        // 2. 矩阵B 与 矩阵C 的 rows 相等时：普通的矩阵减法。
        for(i=0; i<rowB; i++){
            for(j=0; j<colB; j++){
                *(pDataA + i*colB + j) = *(pDataB + i*colB + j) - *(pDataC + i*colB + j);
            }
        }        
    }    
}

/**
 * Update the CKF with the given measurements. On return, self.x
 * and self.P contain the new mean and covariance of the filter.
 * ----------
 * Important: predict() MUST be called before update() is called for the first time.
 */
static void unit_ckf_update(float fz)
{
    unsigned short i=0, j=0;
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
    eyeZero(&zp, sCKF.sm_sigmas_h.numCols, 1); // sm_sigmas_h : 4x1, So the { zp } is 1x1.
    unit_ckf_transform(&sCKF.sm_sigmas_h, &sCKF.smR, &zp, &sCKF.smS);
    //# self.SI = inv(self.S)
    unit_ckf_inverse(&sCKF.smS, &sCKF.smSI); // Get the Inverse.

    //# compute cross variance of the state and the measurements.    
    unsigned short m = sCKF._num_sigmas;
    //# xf = self.x.flatten() // self.x is { sax } 2x1
    //# { self.sigmas_f } is 4x2,       { self.sigmas_h } is 4x1,
    // Pxz = outer_product_sum(self.sigmas_f - xf, self.sigmas_h - zpf) / m
    MATRIX_F32_STRUCT Pxz;
    // 
    unsigned short rowA = sCKF.sm_sigmas_f.numRows;
    unsigned short colA = sCKF.sm_sigmas_f.numCols;
    unsigned short rowB = sCKF.sm_sigmas_h.numRows;
    unsigned short colB = sCKF.sm_sigmas_h.numCols;
    unsigned short rowC = colA, colC = colB;
    float* pLocalDataA = (float*)malloc(rowA*colA*rowB*sizeof(float));
    memset(pLocalDataA, 0.0, (rowA*colA*rowB*sizeof(float)));
    float* pLocalDataB = (float*)malloc(rowA*colA*rowB*sizeof(float));
    memset(pLocalDataB, 0.0, (rowA*colA*rowB*sizeof(float)));
    eyeZero(&Pxz, colA, colB); // Pxz : colA x colB, 2x1

    // Renew ... rows and cols... == self.sigmas_f - xf   ==== 4x2-1x2
    // ===  *(sCKF.sm_sigmas_f.pData + i*rowA + colA) - *(sCKF.saX.pData + colA);
    unit_ckf_matrix_row_subtraction(pLocalDataA, rowA, colA, sCKF.sm_sigmas_f.pData, rowA, colA, sCKF.saX.pData, sCKF.saX.numRows, sCKF.saX.numCols);
    // Renew ... rows and cols... == self.sigmas_h - zpf  === 4x1-1x1
    // === *(sCKF.sm_sigmas_h.pData + i*rowB + colB) - *(zp.pData + colB);
    unit_ckf_matrix_row_subtraction(pLocalDataB, rowB, colB, sCKF.sm_sigmas_h.pData, rowB, colB, zp.pData, zp.numRows, zp.numCols);

    unit_ckf_outer_product_sum(pLocalDataA, rowA, colA, pLocalDataB, rowB, colB, Pxz.pData);
    
    for(i=0; i<rowC*colC; i++){
        *(Pxz.pData + i) /= m;
    }

    //# Kalman gain
    // self.K = dot(Pxz, self.SI)  ---  { 2x1 } = { 2x1 } dot { 1x1 }    
    unit_mat_dot_A_B(&sCKF.smK, &Pxz, &sCKF.smSI);
    
    //# residual
    // self.y = self.residual_z(z, zp)  ---  { 1x1 } = { z 1x1 } - { zp 1x1 }
    unit_ckf_matrix_row_subtraction( sCKF.smY.pData, sCKF.smY.numRows, sCKF.smY.numCols, &fz, 1, 1, zp.pData, zp.numRows, zp.numCols);
    
    //# self.x = self.x + dot(self.K, self.y)  --- { 2x1 } = { 2x1 } + { {2x1} dot {1x1} }
    MATRIX_F32_STRUCT tmpMat, tmpMat2;
    eyeZero(&tmpMat, sCKF.smK.numRows, 1); // Maxtri_2x1
    unit_mat_dot_A_B(&tmpMat, &sCKF.smK, &sCKF.smY);
    unit_mat_add_A_B(&sCKF.saX, &sCKF.saX, &tmpMat);
    free(tmpMat.pData);
        
    // self.P = self.P - dot(self.K, self.S).dot(self.K.T)  ---  { 2x2 } = { 2x2 } - { {{2x1}dot{1x1}} dot{1x2} }    
    eyeZero(&tmpMat, sCKF.smK.numRows, 1); // Maxtri_2x1
    unit_mat_dot_A_B(&tmpMat, &sCKF.smK, &sCKF.smS);    
    eyeZero(&tmpMat2, sCKF.smK.numRows, sCKF.smK.numRows); // Maxtri2_2x2    
    unit_mat_transpose_A_B(&sCKF.smKT, &sCKF.smK); // K 的转置 KT
    unit_mat_dot_A_B(&tmpMat2, &tmpMat, &sCKF.smKT);
    free(tmpMat.pData);
    unit_ckf_matrix_row_subtraction( sCKF.smP.pData, sCKF.smP.numRows, sCKF.smP.numCols, \
        sCKF.smP.pData, sCKF.smP.numRows, sCKF.smP.numCols, tmpMat2.pData, tmpMat2.numRows, tmpMat2.numCols);
    free(tmpMat2.pData);
    
    // # save measurement and posterior state
    *sCKF.smZ.pData = fz;
    memcpy(sCKF.saX_post.pData, sCKF.saX.pData, sCKF.saX_post.numRows*sCKF.saX_post.numCols*sizeof(float));
    memcpy(sCKF.smP_post.pData, sCKF.smP.pData, sCKF.smP_post.numRows*sCKF.smP_post.numCols*sizeof(float));

    // Need to be Free Loval Var zp Memroy !!!
    free(pLocalDataA);
    free(pLocalDataB);
}


/**
 * 计算乘方，乘方就是相同数值的累加。
 * unit_ckf_math_exponent()
 * ----------
 * Returns 乘方 的结果。
 * ----------
 * Parameters
 * nDt : unsigned short
 *     需要被乘方运算的数值。
 * nExp ：unsigned short 
 *     乘方运算的指数。
 */
static float unit_ckf_math_exponent(unsigned short nDt, unsigned short nExp)
{
    unsigned short i = 0;
    float nTmp = nDt;
    
    for(i=1; i<nExp; i++){
        nTmp *= nDt;
    }
    return nTmp;
}

/**
 * Q_discrete_white_noise()
 * ----------
 * Returns the Q matrix for the Discrete Constant White Noise
 * Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
 * is the variance in the noise.
 *     if dim == 2:
 *       Q = [[.25*dt**4, .5*dt**3],
 *            [ .5*dt**3,    dt**2]]
 * ----------
 * Parameters
 * nDim : int (2, 3, or 4)
 *      dimension for Q, where the final dimension is (dim x dim).
 * fVar : float, default=1.0
 *      variance in the noise.
 * smLocalQ : MATRIX_F32_STRUCT *
 *      Maxtri smLocalQ is a MxM .且是一个单位矩阵 E.
 */
static void unit_ckf_Q_discrete_white_noise(MATRIX_F32_STRUCT *smLocalQ, /*unsigned short nDim,*/ unsigned short nDt, float fVar)
{
    if((smLocalQ->numRows != 2) || (smLocalQ->numCols != 2)){
        return; // error
    }
    // Now, regard the smLocalQ as 2x2 .
    *(smLocalQ->pData) = 0.25 * unit_ckf_math_exponent(nDt, 4) * fVar;
    *(smLocalQ->pData+1) = 0.5 * unit_ckf_math_exponent(nDt, 3) * fVar;
    *(smLocalQ->pData+2) = 0.5 * unit_ckf_math_exponent(nDt, 3) * fVar;
    *(smLocalQ->pData+3) = 1.0 * unit_ckf_math_exponent(nDt, 2) * fVar;
}

/**
 * 对外 API ，用于调用 ckf 滤波。
 * ----------
 * Parameters
 * pSampData : short*
 *     [In] U0 或 I0 的原始采样点。
 * nLenSamp : short
 *     [In] U0 或 I0 的原始采样点的数据长度，数据的单位是 szie_t。
 * pSampDataNew : short*
 *     [Out] 滤波后的数据用于判别极性，并且是 float 强制转换为 short，为保留精度，所以扩大10倍。
 */
void unit_ckf_process(short *pSampData , short nLenSamp, /*short*/float *pSampDataNew)
{
    if((NULL == pSampData) || (nLenSamp <= 0) || (NULL == pSampDataNew)){
        return; //error
    }

    unsigned short i=0;
    unsigned short dim_x=2, dim_z=1, dt=2;
    
    unit_ckf_CubatureKalmanFilter(dim_x, dim_z, dt);
    *(sCKF.smR.pData) = 0.5;
    
    MATRIX_F32_STRUCT local_smQ, local_smQ2;
    eyeZero(&local_smQ, 2, 2);
    eyeZero(&local_smQ2, 2, 2);
    memcpy(local_smQ.pData, sCKF.smQ.pData, 4*sizeof(float));
    unit_ckf_Q_discrete_white_noise(&sCKF.smQ, 2, 0.000005);
    memcpy(local_smQ2.pData, sCKF.smQ.pData, 4*sizeof(float));
    unit_mat_AxB(&sCKF.smQ, &local_smQ, &local_smQ2);
    
    float fTmpData = 0.0;
    for(i=0; i<nLenSamp; i++){
        fTmpData = *(pSampData+i);
        unit_ckf_predict();
        unit_ckf_update(fTmpData);
        //*(pSampDataNew+i) = (short)(*(sCKF.saX.pData) * 10);
        *(pSampDataNew+i) = (float)(*(sCKF.saX.pData));
    }
}

/**
 * Test Func.
 * ----------
 */
#if 0
void main(){   

#if 0
    //MATRIX_F32_STRUCT local_smU;
    //eyeZero(&local_smU, 3, 3); 
			
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

    // Test 
    MATRIX_F32_STRUCT smTmpXs;
    unsigned short size = sCKF.saX.numCols*sCKF.saX.numRows;
    eyeZero(&smTmpXs, size, size);
    unit_ckf_matrix_outer(sCKF.saX.pData, size, &smTmpXs);

    // Test --- unit_ckf_update()
    for(unsigned short i=0; i<sCKF.sm_sigmas_f.numRows*sCKF.sm_sigmas_f.numCols; i++){
       *(sCKF.sm_sigmas_f.pData+i) = i;
    }
    // unit_ckf_update();

    // Test --- unit_ckf_inverse() --- [Begin] ok
    //for(unsigned short i=0; i<sCKF.smS.numRows*sCKF.smS.numCols; i++){
    //  *(sCKF.smS.pData+i) = i+1;
    //}
    //for(unsigned short i=0; i<local_smU.numRows*local_smU.numCols; i++){
    //  *(local_smU.pData+i) = i+1;
    //}
    MATRIX_F32_STRUCT local_smU;
    eyeZero(&local_smU, 2, 2);
    *(local_smU.pData) = 5;    
	*(local_smU.pData+1) = 1;
	*(local_smU.pData+2) = 2;
	*(local_smU.pData+3) = 4;
	//*(local_smU.pData+4) = 5;
	//*(local_smU.pData+5) = 6;
	//*(local_smU.pData+6) = 1;
	//*(local_smU.pData+7) = 8;
	//*(local_smU.pData+8) = 9;
    MATRIX_F32_STRUCT local_smSI;
    eyeZero(&local_smSI, 2, 2);
    unit_ckf_inverse(&local_smU, &local_smSI); //&sCKF.smSI); // Get the Inverse.
    // Test --- unit_ckf_inverse() --- [End] ok

    // Test --- unit_ckf_outer_product_sum() --- [Begin]    
    MATRIX_F32_STRUCT local_smUA;
    eyeZero(&local_smUA, 2, 2);
    *(local_smUA.pData) = 9;
	*(local_smUA.pData+1) = 1;
	*(local_smUA.pData+2) = 2;
	*(local_smUA.pData+3) = 4;
	//*(local_smUA.pData+4) = 5;
	//*(local_smUA.pData+5) = 6;
	//*(local_smUA.pData+6) = 1;
	//*(local_smUA.pData+7) = 8;
    MATRIX_F32_STRUCT local_smUB;
    eyeZero(&local_smUB, 1, 2);
    *(local_smUB.pData) = 9;
	*(local_smUB.pData+1) = 21;
	//*(local_smUB.pData+2) = 12;
	//*(local_smUB.pData+3) = 2;
    MATRIX_F32_STRUCT local_smUC;
    //eyeZero(&local_smUC, local_smUA.numCols, local_smUB.numCols); // local_smUC : colA x colB
    //unit_ckf_outer_product_sum(local_smUA.pData, 4, 2, local_smUB.pData, 4, 1, local_smUC.pData);
    // Test --- unit_ckf_outer_product_sum() --- [End]

    // Test --- unit_ckf_matrix_row_subtraction() --- [Begin]
    eyeZero(&local_smUC, 1, 1); // local_smUC : 4 x 2
    float A=5.0, B=9.0;
    unit_ckf_matrix_row_subtraction(local_smUC.pData, 1, 1, &A, &B);
    //unit_ckf_matrix_row_subtraction(local_smUB.pData, 2, 2, local_smUB.pData, 2, 2, local_smUA.pData, 2, 2);
    //unit_ckf_matrix_row_subtraction(local_smUA.pData, 2, 2, local_smUA.pData, 2, 2, local_smUB.pData, 1, 2);
    // Test --- unit_ckf_matrix_row_subtraction() --- [End]

    // Test --- unit_mat_dot_A_B() --- [Begin]
    eyeZero(&local_smUC, 4, 2); // local_smUC : 4 x 2
    unit_mat_dot_A_B(&local_smUC, &local_smUA, &local_smUB);
    // Test --- unit_mat_dot_A_B() --- [End]

    // Test --- unit_mat_add_A_B() --- [Begin]
    eyeZero(&local_smUC, 2, 2); // local_smUC : 2 x 2
    unit_mat_add_A_B(&local_smUC, &local_smUA, &local_smUB);
    // Test --- unit_mat_add_A_B() --- [End]

    // Test --- unit_mat_transpose_A_B() --- [Begin]
    eyeZero(&local_smUC, 2, 2); // local_smUC : 2 x 2
    unit_mat_transpose_A_B( &local_smUC, &local_smUB);
    // Test --- unit_mat_transpose_A_B() --- [End]
    
    // Test --- unit_ckf_math_exponent() --- [Begin]
    float tmp = unit_ckf_math_exponent(7, 6);
    // Test --- unit_ckf_math_exponent() --- [End]
    
    // Test --- float to short --- [Begin]
    short nA = 10;
    float fB = 12.645; 
    printf("First	nA=%d \r\n", nA);
    printf("Second	fB=%f \r\n", fB);
    nA = (short)fB;
    printf("Second	nA=%d \r\n", nA);
    // Test --- float to short --- [End]
#endif

    short nSampData[] = {5, -8, -15, -2, 1, -12, -17, 8, 21, 12, -6, -15, -18, -9, 2, -9, 3, -1, -6, 5, -6, -15, -7, 3, 5, 7, 0, -12, -16, 1, -4, -14, -14, 2, 6, -2, -13, -15, 2, 4, 1, -5, 21, 17, 19, 29, 15, 26, 10, 6, 6, -8, 4, -3, -13, -15, 2, 5, 6, 7, 6, -6, 2, 5, 7, 8, 7, 8, 7, 7, -6, -14, -5, 4, 5, -6, 10, 21, -2, 7, 8, 7, 8, 8, 8, 8, 4, -8, 4, -4, 0, 6, 7, 7, 8, 8, 6, -8, 4, -1, -4, 4, 6, -3, 4, 23, 0, 6, -2, -2, 5, 6, -3, 0, 5, 6, 7, 7, 3, -8, 5, 6, 8, 8, 7, 8, 8, 11, 26, 1, 6, 0, -4, 6, 6, 8, 8, 7, 7, 20, 22, 13, 10, 23, 21, 13, 8, -8, 4, 6, 10, 26, 17, 11, 9, 6, -9, -16, -18, -8, 2, 4, 7, 7, 5, -10, -16, -19, -6, 2, -10, -17, -2, 4, 6, -7, -15, -4, 2, -10, -17, -20, -19, -17, 15, 6, -9, -9, 3, -8, -15, -6, 4, 5, 6, 7, 8, 7, 8, -2, 0, 4, -9, 3, -2, -3, 4, -8, 2, 0, -13, -17, 0, 4, 10, 26, 17, 0, 2, 5, 6, 8, 17, 20, 0, 7, -4, -15, -18, -19, 0, 4, 0, -7, 4, 6, 6, 7, 8, 8, -4, 0, 6, 23, 32, 27, 15, 11, 8, -3, -14, -10, 3, -8, -15, -19, -16, 1, 5, 7, 7, 7, 8, 6, 1, -6, 4, -6, -15, -9, 2, 6, 7, 7, 8, 8, 7, -1, -1, 6, 7, -3, 1, 7, 25, 4, 6, 2, -11, -17, 0, -3, -13, -17, 1, 5, 1, -12, -17, 0, 4, 2, -9, 4, -4, -14, -18, -20, 0, 4, 6, 6, -8, 15, 22, 13, 9, 9, 8, 7, 8, 8, -4, -14, -10, 2, -8, -15, -6, 1, -10, -17, -19, -10, 2, 5, -4, 0, 6, 24, 19, 20, 28, 16, 10, -2, 0, 7, 25, 19, 2, 0, 5, 7, -5, -15, -10, 2, 6, -4, -13, -12, 1, -8, 0, 5, 8, 7, -3, 0, 6, 7, -6, 2, 1, -11, -16, -19, -4, 4, 6, 6, 8, 8, 7, -8, -16, -2, 3, 7, 7, -3, 0, 5, 6, -5, -15, -8, 3, 5, 6, 1, -6, 4, 6, 8, 8, 8, 8, -4, 1, 6, 8, 6, 6, 2, -10, -17, 0, -2, -13, -17, 0, 5, 6, 6, -8, 4, 6, 4, -10, -17, 0, 4, 4, -9, 4, 5, 3, -8, 4, -5, 0, 4, 7, 7, -2, -13, -15, 2, -7, 0, 2, -9, 3, -4, -1, 4, -9, -17, -19, -10, 2, -9, -17, -18, -15, 1, -7, -17, -18, -19, -20, -1, -3, -13, -18, -19, -19, -20, -20, -21, -2, -10, -33, -26, -23, -21, -21, -21, -20, -1, 4, 6, 7, -7, 13, 23, 13, 10, -4, 2, 5, 7, 8, 19, 23, 12, -6, 5, 6, 2, -8, 4, 6, 7, 7, 8, -6, 11, 23, 15, 10, 8, 8, 5, -8, 4, 6, 6, 6, -10, -15, -2, 0, -10, 4, 6, 6, 8, 8, 7, 8, 3, -8, 4, 6, 0, -5, 4, 6, -3, -13, -13, 2, 4, 6, 4, -9, 4, 6, 2, -6, 4, 6, 6, 4, -8, 4, 6, 6, 5, -10, 3, -3, -2, 4, 6, 8, 0, -13, -17, 1, -4, -2, 4, 6, -5, 1, 5, 6, -8, 3, 0, -13, -17, 0, -4, -3, 4, 6, 6, 7, 8, 7, 21, 23, 13, 9, -7, -15, -18, -12, 2, 6, 6, 6, 6, -8, -16, -2, 0, -10, 4, 6, 7, 8, 7, -5, -15, 67, 527, 4887, 2306, -290, 1991, -939, -177, 548, 769, 1740, -702, -700, 287, 867, -141, 1021, 592, -606, -525, 514, 752, 559, 247, 70, -94, 14, 403, 1006, 352, -70, -52, 201, 290, 555, 507, 227, -129, 3, 384, 528, 443, 275, 80, 36, 181, 484, 501, 258, 108, 82, 186, 356, 429, 358, 177, 61, 180, 312, 386, 356, 232, 130, 138, 242, 373, 368, 256, 142, 106, 201, 290, 305, 252, 145, 145, 199, 262, 305, 276, 188, 134, 136, 221, 274, 247, 190, 138, 148, 214, 254, 250, 195, 155, 151, 179, 222, 239, 199, 163, 146, 154, 190, 222, 212, 179, 166, 177, 195, 208, 211, 181, 166, 160, 166, 196, 193, 177, 164, 162, 177, 214, 214, 193, 171, 162, 172, 189, 206, 213, 201, 627, 592, -358, 420, 189, 83, 239, 229, 254, 212, 121, 58, 374, 220, 184, 266, 303, 114, 191, 279, 273, 223, 202, 202, 212, 214, 216, 270, 265, 199, 173, 224, 235, 230, 241, 247, 220, 209, 220, 254, 238, 225, 227, 238, 226, 235, 244, 242, 226, 206, 206, 219, 237, 245, 231, 212, 201, 210, 232, 241, 245, 243, 229, 206, 205, 211, 212, 198, 203, 220, 229, 205, 208, 212, 208, 194, 189, 200, 201, 192, 187, 186, 186, 177, 167, 194, 177, 173, 180, 183, 184, 184, 184, 182, 164, 158, 156, 158, 174, 180, 168, 151, 138, 145, 150, 151, 151, 154, 170, 179, 172, 160, 156, 153, 152, 145, 134, 144, 149, 151, 151, 152, 152, 165, 177, 177, 164, 158, 153, 152, 151, 152, 166, 177, 181, 183, 182, 543, 397, -474, 316, 150, 86, 211, 232, 256, 159, 82, 115, 328, 146, 193, 288, 140, 156, 238, 251, 195, 186, 186, 184, 189, 207, 263, 208, 173, 185, 220, 203, 202, 202, 189, 167, 201, 238, 247, 211, 209, 212, 212, 217, 249, 221, 208, 211, 211, 212, 225, 229, 217, 197, 203, 209, 211, 210, 195, 188, 184, 186, 200, 193, 197, 193, 168, 157, 182, 182, 171, 194, 190, 184, 175, 163, 172, 163, 155, 157, 171, 159, 154, 151, 151, 150, 149, 149, 149, 149, 149, 149, 149, 149, 148, 149, 149, 143, 130, 141, 145, 147, 149, 149, 149, 149, 142, 131, 140, 130, 123, 121, 120, 119, 119, 119, 121, 137, 144, 137, 131, 141, 146, 147, 149, 148, 148, 134, 379, 201, -581, 283, 128, 64, 197, 207, 242, 140, 78, 128, 331, 133, 173, 270, 98, 106, 203, 237, 183, 163, 154, 162, 173, 181, 231, 175, 138, 150, 185, 185, 204, 210, 175, 176, 206, 219, 206, 173, 174, 177, 186, 203, 226, 206, 179, 169, 193, 203, 195, 185, 180, 162, 169, 185, 195, 185, 163, 169, 175, 181, 197, 189, 194, 197, 185, 164, 169, 157, 141, 162, 156, 174, 182, 162, 152, 149, 138, 130, 140, 128, 134, 134, 124, 119, 117, 128, 134, 125, 136, 143, 138, 128, 138, 128, 121, 124, 136, 125, 121, 126, 135, 123, 119, 129, 132, 125, 137, 128, 131, 131, 104, 126, 126, 121, 120, 135, 127, 108, 100, 110, 130, 127, 121, 121, 136, 127, 131, 140, 144, 145, 146, 156, 164, 154, 149, 147, 147, 147, 147, 146, 146, 147, 145, 146, 145, 146, 148, 165, 157, 162, 167, 156, 167, 173, 176, 177, 177, 193, 192, 184, 178, 160, 151, 158, 170, 174, 177, 177, 184, 195, 167, 169, 175, 177, 177, 160, 164, 164, 156, 169, 173, 165, 153, 148, 147, 145, 154, 164, 154, 148, 158, 164, 153, 148, 146, 145, 145, 145, 145, 145, 138, 124, 118, 130, 130, 121, 117, 116, 114, 115, 114, 114, 114, 125, 126, 101, 108, 111, 114, 114, 114, 98, 100, 103, 91, 86, 97, 99, 93, 106, 110, 100, 88, 85, 82, 97, 108, 111, 112, 96, 88, 92, 103, 91, 101, 109, 112, 114, 114, 113, 114, 114, 114, 98, 110, 125, 119, 116, 114, 114, 114, 114, 130, 139, 141, 143, 144, 144, 144, 144, 144, 143, 145, 144, 144, 144, 145, 155, 168, 173, 174, 162, 160, 166, 155, 164, 159, 150, 147, 162, 156, 158, 169, 173, 175, 162, 157, 167, 154, 176, 185, 175, 156, 149, 158, 162, 151, 146, 145, 155, 162, 151, 147, 145, 144, 140, 125, 133, 128, 125, 134, 123, 117, 114, 113, 114, 129, 125, 118, 117, 131, 109, 102, 109, 112, 112, 112, 105, 93, 103, 94, 85, 88, 101, 91, 98, 99, 88, 83, 82, 81, 81, 82, 99, 107, 101, 93, 104, 93, 85, 82, 81, 81, 95, 97, 86, 82, 82, 80, 81, 83, 100, 107, 100, 88, 83, 82, 94, 90, 73, 95, 89, 84, 82, 82, 99, 106, 118, 129, 119, 131, 114, 100, 110, 128, 123, 125, 136, 140, 142, 155, 159, 149, 145, 128, 119, 120, 132, 122, 129, 136, 139, 141, 142, 131, 120, 117, 130, 137, 141, 140, 141, 142, 142, 142, 143, 161, 154, 147, 144, 143, 142, 141, 142, 141, 142, 142, 141, 141, 143, 159, 154, 146, 141, 123, 117, 124, 128, 119, 114, 97, 105, 123, 117, 113, 122, 134, 137, 123, 129, 118, 99, 105, 76, 71, 75, 80, 97, 91, 82, 81, 80, 80, 79, 71, 61, 71, 60, 64, 73, 75, 78, 78, 78, 78, 79, 80, 78, 86, 97, 71, 71, 75, 78, 78, 78, 67, 62, 73, 76, 78, 78, 78, 78, 78, 78, 71, 62, 90, 86, 81, 80, 80, 79, 93, 95, 84, 81, 80, 91, 103, 107, 108, 110, 111, 110, 109, 110, 98, 93, 104, 108, 110, 119, 132, 137, 138, 139, 132, 120, 114, 126, 135, 134, 119, 114, 125, 127, 116, 112, 111, 110, 110, 110, 110, 123, 133, 134, 120, 113, 112, 110, 111, 128, 120, 114, 111, 110, 110, 122, 124, 116, 112, 111, 110, 110, 109, 91, 84, 80, 82, 98, 88, 81, 86, 100, 106, 93, 84, 80, 79, 95, 91, 90, 99, 86, 68, 65, 69, 56, 67, 60, 60, 68, 57, 65, 73, 74, 75, 60, 52, 49, 52, 67, 73, 75, 77, 74, 59, 52, 58, 66, 55, 50, 48, 48, 47, 47, 47, 57, 69, 73, 60, 52, 49, 47}; 
	short nLen = sizeof(nSampData)/sizeof(short);
	//short *pfSamp = (short *)malloc(nLen*sizeof(short));
	//memset(pfSamp, 0, nLen*sizeof(short));
	float *pfSamp = (float *)malloc(nLen*sizeof(float));
	memset(pfSamp, 0, nLen*sizeof(float));
    unit_ckf_process(nSampData, nLen, pfSamp); 
	
	return;    
}
#endif
