
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//#include "Filterpy_UNIT_AlgorithmCKF.h"
//#include "Globevar.h" 

// Private Var

/**
 * 
 */
/// @brief Use Memroy in SDRAM for Global Matrix, Instead of malllc()
#define MEM_AREA_SIZE 128
float gfMemArea[MEM_AREA_SIZE];
unsigned short gMemAreaPos = 0; // Record the starter Pos of rest area [] .

/// @brief Use Memroy in SDRAM for Tmp Matrix, Instead of malllc(), And Need Free Mem Manully.
#define TEMP_MEM_AREA_SIZE 64
typedef struct
{
    float gfMemTmp[TEMP_MEM_AREA_SIZE];
    unsigned short gMemTmpPos; // Record the starter Pos of rest area A[] .
} MEM_TMP_F32_STRUCT;
static MEM_TMP_F32_STRUCT gsMemTmpA;
static MEM_TMP_F32_STRUCT gsMemTmpB;
static MEM_TMP_F32_STRUCT gsMemTmpC;

/**
 * @brief Instance structure for the floating-point matrix structure.
 */
typedef struct
{
    unsigned short numRows;     /**< number of rows of the matrix.     */
    unsigned short numCols;     /**< number of columns of the matrix.  */
	unsigned short numPos;		/**< points Position to the data of the matrix  IN gfMemArea[] . */	
    float *pData;     			/**< points to the data of the matrix. */
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
static void unit_ckf_predict(void);

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
	unsigned short i=0;
	for(i=0; i<128 /*MEM_AREA_SIZE*/; i++){
       gfMemArea[i] = i;
	}	

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
 * ----------
 * @2018-12-26 Used for Instead of malloc(), For Temp Matrix.
 * ----------
 * Parameters
 * : Num of float Var, In this time want to malloc() .
 * ：Last Position in this [] .
 * : The rest of Num of float Var, or Tolal Num of float Var in this [] .
 * ：Return, the Position float Pointer in this [] .
 */ 
static float* unit_mat_malloc_tmp_memroy(unsigned short nNum, MEM_TMP_F32_STRUCT *gsMemTmp)
{
	if((nNum + gsMemTmp->gMemTmpPos) >=  64){
		return NULL; // fatal error
	}
	
	unsigned short nLastPos = gsMemTmp->gMemTmpPos;
	unsigned short nCount = nNum;
	
	do{
		gsMemTmp->gfMemTmp[gsMemTmp->gMemTmpPos] = 0.0f;
		gsMemTmp->gMemTmpPos++;
		nCount--;
	} while (nCount > 0);

    // Matrix Data Address
	return (&gsMemTmp->gfMemTmp[nLastPos]);
}

static void unit_mat_free_tmp_memroy(unsigned short nNum, MEM_TMP_F32_STRUCT *gsMemTmp)
{
    //if(nNum != gsMemTmp->gMemTmpPos){
	//	return ; // fatal error
	//}

    gsMemTmp->gMemTmpPos = 0; // Clear Pos, and return to the Starter of  A[]
    
    return;
}

/**
 * ----------
 * @2018-12-25 Used for Instead of malloc(), For Global Matrix.
 * ----------
 * Parameters
 * : Num of float Var, In this time want to malloc() .
 * ：Last Position in this [] .
 * : The rest of Num of float Var, or Tolal Num of float Var in this [] .
 * ：Return, the Position float Pointer in this [] .
 */ 
static float* unit_mat_malloc_memroy(unsigned short nNum)
{
	if((nNum + gMemAreaPos) >=  128){
		return NULL; // fatal error
	}
	
	unsigned short nLastPos = gMemAreaPos;
	unsigned short nCount = nNum;
	
	do{
		gfMemArea[gMemAreaPos] = 0.0f;
		gMemAreaPos++;
		nCount--;
	} while (nCount > 0);

    // Matrix Data Address
	return &gfMemArea[nLastPos];
}

/**
 * @2018-12-15 Later, Need to check MEMORY.
 */ 
static void unit_mat_zero_f32(MATRIX_F32_STRUCT* s)
{
	unsigned short pos = 0;
	unsigned short blockSize = s->numRows * s->numCols;

	s->pData = unit_mat_malloc_memroy(blockSize);
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
    //eyeZero(&local_smU, nRows, nRows);
    local_smU.numRows = nRows;
    local_smU.numCols = nRows; // nRows == nCols
    local_smU.pData = (float*)unit_mat_malloc_tmp_memroy(nRows*nRows, &gsMemTmpA);
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
    // Clear Memory    
    unit_mat_free_tmp_memroy(nRows*nRows, &gsMemTmpA);
    local_smU.pData = NULL;
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
static void unit_ckf_predict(void)
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
    //eyeZero(&smTmpXs, size, size);
    //eyeZero(&smTmpxf, size, size);
    smTmpXs.numRows = smTmpXs.numCols = size;    
    smTmpXs.pData = unit_mat_malloc_tmp_memroy(size*size, &gsMemTmpA);
    smTmpxf.numRows = smTmpxf.numCols = size;    
    smTmpxf.pData = unit_mat_malloc_tmp_memroy(size*size, &gsMemTmpB);

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
    unit_mat_free_tmp_memroy(size*size, &gsMemTmpA);
    smTmpXs.pData = NULL;
    unit_mat_free_tmp_memroy(size*size, &gsMemTmpB);
    smTmpxf.pData = NULL;
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
    //float* pLocalData = (float*)malloc(rowA*colA*colB*sizeof(float));
    float * pLocalData = unit_mat_malloc_tmp_memroy(rowA*colA*colB, &gsMemTmpA);
    memset(pLocalData, 0.0, (rowA*colA*colB*sizeof(float)));
    float* pTmpData = pLocalData; // Used for Move pointer, Now is Header.
    
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
    unit_mat_free_tmp_memroy(rowA*colA*colB, &gsMemTmpA);
    pTmpData = NULL;
	pLocalData = NULL;
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
    float fzTmp = fz;
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
    //eyeZero(&zp, sCKF.sm_sigmas_h.numCols, 1); // sm_sigmas_h : 4x1, So the { zp } is 1x1. @2018-12-26 固定 Data 大小，不在调用 eyeZero()，避免内存管理。
    zp.numRows = 1;
    zp.numCols = 1;
    zp.pData = unit_mat_malloc_tmp_memroy(zp.numRows*zp.numCols, &gsMemTmpC);
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
    unsigned short rowA = sCKF.sm_sigmas_f.numRows; // 4
    unsigned short colA = sCKF.sm_sigmas_f.numCols; // 2
    unsigned short rowB = sCKF.sm_sigmas_h.numRows; // 4
    unsigned short colB = sCKF.sm_sigmas_h.numCols; // 1
    unsigned short rowC = colA, colC = colB;
    //float* pLocalDataA = (float*)malloc(rowA*colA*rowB*sizeof(float));
    float pLocalDataA[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // rowA*colA*rowB==4*2*1==8
    //memset(&pLocalDataA[0], 0, (rowA*colA*rowB*sizeof(float)));
    //float* pLocalDataB = (float*)malloc(rowA*colA*rowB*sizeof(float));
    float pLocalDataB[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // rowA*colA*rowB==4*2*1==8
    //memset(&pLocalDataB[0], 0, (rowA*colA*rowB*sizeof(float)));
    //eyeZero(&Pxz, colA, colB); // Pxz : colA x colB, 2x1
    Pxz.numRows = colA;
    Pxz.numCols = colB;
    Pxz.pData = unit_mat_malloc_tmp_memroy(Pxz.numRows*Pxz.numCols, &gsMemTmpA);

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
    // Clear Memory of { Pzx }    
    unit_mat_free_tmp_memroy(Pxz.numRows*Pxz.numCols, &gsMemTmpA);
    Pxz.pData = NULL;

    //# residual
    // self.y = self.residual_z(z, zp)  ---  { 1x1 } = { z 1x1 } - { zp 1x1 }
    unit_ckf_matrix_row_subtraction( sCKF.smY.pData, sCKF.smY.numRows, sCKF.smY.numCols, &fzTmp, 1, 1, zp.pData, zp.numRows, zp.numCols);
    
    //# self.x = self.x + dot(self.K, self.y)  --- { 2x1 } = { 2x1 } + { {2x1} dot {1x1} }
    MATRIX_F32_STRUCT tmpMat, tmpMat2;
    //eyeZero(&tmpMat, sCKF.smK.numRows, 1); // Maxtri_2x1
    {
        tmpMat.numRows = sCKF.smK.numRows;
        tmpMat.numCols = 1;
        tmpMat.pData = unit_mat_malloc_tmp_memroy(tmpMat.numRows*tmpMat.numCols, &gsMemTmpA);
    }
    unit_mat_dot_A_B(&tmpMat, &sCKF.smK, &sCKF.smY);
    unit_mat_add_A_B(&sCKF.saX, &sCKF.saX, &tmpMat);
    {
        // Clear Memory of { tmpMat }        
        unit_mat_free_tmp_memroy(tmpMat.numRows*tmpMat.numCols, &gsMemTmpA);
        tmpMat.pData = NULL;
    }
        
    // self.P = self.P - dot(self.K, self.S).dot(self.K.T)  ---  { 2x2 } = { 2x2 } - { {{2x1}dot{1x1}} dot{1x2} }    
    //eyeZero(&tmpMat, sCKF.smK.numRows, 1); // Maxtri_2x1
    {
        tmpMat.numRows = sCKF.smK.numRows;
        tmpMat.numCols = 1;
        tmpMat.pData = unit_mat_malloc_tmp_memroy(tmpMat.numRows*tmpMat.numCols, &gsMemTmpA);
    }
    unit_mat_dot_A_B(&tmpMat, &sCKF.smK, &sCKF.smS);    
    //eyeZero(&tmpMat2, sCKF.smK.numRows, sCKF.smK.numRows); // Maxtri2_2x2    
    {
        tmpMat2.numRows = sCKF.smK.numRows;
        tmpMat2.numCols = sCKF.smK.numRows;
        tmpMat2.pData = unit_mat_malloc_tmp_memroy(tmpMat2.numRows*tmpMat2.numCols, &gsMemTmpB);
    }
    unit_mat_transpose_A_B(&sCKF.smKT, &sCKF.smK); // K 的转置 KT
    unit_mat_dot_A_B(&tmpMat2, &tmpMat, &sCKF.smKT);
    //free(tmpMat.pData);
    unit_ckf_matrix_row_subtraction( sCKF.smP.pData, sCKF.smP.numRows, sCKF.smP.numCols, \
        sCKF.smP.pData, sCKF.smP.numRows, sCKF.smP.numCols, tmpMat2.pData, tmpMat2.numRows, tmpMat2.numCols);
    
    // Need to be Free Memroy !!!    
    unit_mat_free_tmp_memroy(tmpMat.numRows*tmpMat.numCols, &gsMemTmpA);
    tmpMat.pData = NULL;
    unit_mat_free_tmp_memroy(tmpMat2.numRows*tmpMat2.numCols, &gsMemTmpB);
    tmpMat2.pData = NULL;
    unit_mat_free_tmp_memroy(tmpMat.numRows*tmpMat.numCols, &gsMemTmpC);
    zp.pData = NULL;

    // # save measurement and posterior state
    *sCKF.smZ.pData = fz;
    memcpy(sCKF.saX_post.pData, sCKF.saX.pData, sCKF.saX_post.numRows*sCKF.saX_post.numCols*sizeof(float));
    memcpy(sCKF.smP_post.pData, sCKF.smP.pData, sCKF.smP_post.numRows*sCKF.smP_post.numCols*sizeof(float));
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
    //*(sCKF.smP.pData) = 33.5;
    *(sCKF.smR.pData) = 0.5;
    
    MATRIX_F32_STRUCT local_smQ, local_smQ2;
    //eyeZero(&local_smQ, 2, 2);
    local_smQ.numRows = 2;
    local_smQ.numCols = 2;
    local_smQ.pData = unit_mat_malloc_tmp_memroy(local_smQ.numRows*local_smQ.numCols, &gsMemTmpA);
    //eyeZero(&local_smQ2, 2, 2);
    local_smQ2.numRows = 2;
    local_smQ2.numCols = 2;
    local_smQ2.pData = unit_mat_malloc_tmp_memroy(local_smQ2.numRows*local_smQ2.numCols, &gsMemTmpB);
    memcpy(local_smQ.pData, sCKF.smQ.pData, 4*sizeof(float));
    unit_ckf_Q_discrete_white_noise(&sCKF.smQ, 2, 0.0005);
    memcpy(local_smQ2.pData, sCKF.smQ.pData, 4*sizeof(float));
    unit_mat_AxB(&sCKF.smQ, &local_smQ, &local_smQ2);
    // Clear Tmp Memory
    unit_mat_free_tmp_memroy(local_smQ.numRows*local_smQ.numCols, &gsMemTmpA);
    local_smQ.pData = NULL;
    unit_mat_free_tmp_memroy(local_smQ2.numRows*local_smQ2.numCols, &gsMemTmpB);
    local_smQ2.pData = NULL;

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
 * Test Case Func.
 * ----------
 * Output : Hello, world.
 */
void unit_ckf_test_case()
{
    printf("Hello, world. This is C code.\r\n");
}

/**
 * Test Func.
 * ----------
 */
#if 1
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

    short nSampData[] = {34, 33, 34, 33, 34, 33, 34, 33, 34, 34, 34, 34, 34, 33, 34, 34, 34, 34, 33, 34, 33, 33, 34, 33, 33, 34, 33, 34, 33, 34, 33, 33, 33, 34, 33, 34, 34, 33, 34, 34, 34, 34, 33, 34, 34, 33, 34, 34, 34, 33, 34, 33, 35, 33, 33, 34, 34, 34, 33, 34, 33, 34, 34, 34, 35, 33, 34, 34, 34, 34, 33, 34, 34, 34, 32, 33, 34, 33, 34, 34, 33, 35, 33, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 34, 34, 33, 33, 33, 33, 33, 34, 33, 33, 33, 34, 34, 34, 33, 33, 33, 34, 33, 34, 34, 34, 33, 33, 34, 34, 33, 34, 33, 34, 33, 34, 34, 34, 34, 35, 34, 34, 33, 33, 34, 33, 34, 33, 34, 33, 34, 33, 33, 34, 33, 34, 33, 34, 34, 34, 34, 33, 34, 35, 33, 33, 34, 34, 34, 33, 33, 33, 33, 33, 34, 34, 33, 33, 33, 33, 34, 34, 33, 33, 33, 33, 34, 34, 34, 33, 34, 34, 34, 33, 33, 33, 34, 33, 33, 34, 33, 34, 34, 34, 33, 33, 33, 34, 34, 33, 33, 33, 34, 33, 34, 32, 33, 33, 34, 33, 34, 33, 33, 34, 34, 34, 33, 34, 34, 33, 33, 33, 34, 34, 34, 33, 34, 34, 33, 33, 33, 33, 34, 34, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 33, 33, 33, 34, 34, 34, 34, 34, 33, 33, 33, 33, 34, 34, 33, 34, 33, 34, 33, 33, 33, 34, 33, 33, 33, 34, 34, 34, 33, 33, 34, 33, 34, 34, 34, 34, 33, 34, 33, 34, 33, 33, 34, 33, 34, 33, 34, 34, 34, 33, 34, 34, 34, 34, 33, 33, 34, 34, 34, 33, 34, 33, 34, 33, 34, 34, 34, 34, 33, 33, 33, 33, 34, 34, 34, 34, 33, 34, 33, 34, 34, 34, 34, 35, 34, 34, 34, 34, 34, 34, 33, 34, 34, 33, 33, 33, 33, 34, 34, 33, 34, 34, 34, 35, 33, 34, 33, 34, 34, 33, 33, 34, 33, 34, 34, 33, 33, 34, 35, 34, 34, 34, 34, 34, 34, 34, 33, 35, 33, 33, 34, 34, 34, 34, 33, 33, 34, 33, 34, 33, 33, 34, 34, 34, 35, 34, 33, 34, 34, 33, 34, 34, 34, 34, 34, 33, 34, 33, 33, 33, 33, 33, 34, 34, 33, 34, 34, 33, 34, 34, 34, 33, 33, 34, 33, 33, 34, 33, 34, 34, 34, 34, 34, 33, 34, 34, 34, 33, 34, 33, 34, 33, 34, 33, 34, 33, 34, 34, 34, 34, 34, 34, 34, 34, 33, 34, 34, 33, 34, 33, 33, 33, 34, 33, 34, 33, 34, 34, 33, 34, 34, 33, 33, 34, 34, 34, 34, 34, 33, 33, 33, 34, 33, 33, 33, 33, 34, 33, 33, 33, 34, 35, 33, 34, 33, 34, 33, 34, 34, 34, 34, 34, 33, 33, 34, 30, 30, 31, 32, 31, 31, 31, 31, 32, 32, 32, 33, 33, 34, 34, 34, 35, 35, 34, 35, 34, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 37, 37, 37, 37, 37, 38, 36, 36, 36, 36, 36, 36, 35, 35, 35, 36, 35, 34, 34, 34, 34, 34, 33, 33, 32, 32, 33, 32, 32, 31, 31, 31, 31, 31, 30, 31, 30, 31, 29, 29, 29, 30, 29, 29, 30, 30, 29, 29, 29, 29, 29, 30, 30, 30, 30, 29, 30, 30, 30, 30, 31, 31, 32, 31, 32, 32, 32, 33, 33, 33, 34, 33, 33, 34, 34, 34, 35, 35, 36, 35, 35, 36, 36, 36, 36, 36, 37, 36, 37, 36, 37, 38, 37, 38, 38, 38, 37, 37, 37, 38, 37, 37, 37, 38, 38, 36, 37, 37, 36, 36, 36, 36, 36, 36, 35, 35, 35, 36, 34, 34, 35, 33, 34, 34, 33, 33, 33, 32, 32, 32, 31, 31, 30, 30, 31, 31, 31, 31, 30, 30, 30, 29, 30, 29, 29, 30, 30, 29, 30, 29, 30, 29, 30, 29, 30, 30, 30, 31, 32, 31, 31, 32, 31, 32, 32, 32, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 35, 37, 37, 37, 37, 37, 38, 37, 37, 38, 38, 38, 37, 38, 38, 37, 37, 37, 37, 37, 37, 37, 37, 36, 36, 36, 35, 35, 34, 34, 35, 34, 34, 34, 33, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 30, 30, 30, 31, 29, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 29, 29, 30, 30, 31, 30, 31, 31, 30, 31, 31, 31, 32, 31, 32, 32, 32, 32, 33, 34, 33, 33, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 37, 38, 38, 38, 37, 37, 37, 38, 37, 37, 37, 37, 37, 37, 36, 36, 36, 35, 36, 35, 35, 33, 35, 34, 33, 34, 33, 33, 33, 33, 33, 32, 31, 31, 30, 31, 31, 31, 30, 30, 30, 30, 31, 30, 30, 30, 30, 30, 29, 29, 30, 30, 31, 29, 31, 30, 30, 30, 30, 31, 30, 30, 31, 30, 31, 31, 31, 32, 31, 32, 32, 33, 33, 33, 35, 34, 34, 35, 34, 34, 36, 36, 35, 36, 35, 35, 37, 36, 36, 36, 37, 37, 37, 38, 38, 37, 38, 38, 39, 37, 37, 38, 38, 37, 37, 37, 37, 38, 38, 36, 36, 37, 36, 36, 36, 36, 35, 35, 35, 35, 35, 34, 35, 35, 34, 34, 33, 33, 33, 33, 32, 32, 32, 31, 31, 31, 30, 31, 31, 30, 31, 30, 29, 30, 30, 30, 30, 29, 30, 30, 29, 30, 29, 29, 30, 29, 30, 30, 30, 31, 30, 31, 31, 31, 31, 31, 32, 33, 32, 33, 33, 32, 33, 33, 34, 33, 33, 35, 35, 34, 35, 35, 35, 36, 35, 37, 36, 36, 37, 37, 37, 37, 37, 38, 37, 37, 37, 38, 38, 37, 38, 37, 37, 37, 38, 38, 38, 37, 37, 38, 37, 37, 37, 37, 36, 36, 36, 36, 35, 35, 35, 35, 34, 34, 34, 34, 33, 33, 33, 32, 33, 32, 31, 32, 31, 31, 31, 30, 31, 30, 30, 30, 30, 30, 29, 29, 29, 31, 29, 29, 30, 30, 29, 30, 30, 29, 29, 30, 30, 30, 30, 29, 31, 31, 31, 31, 32, 32, 32, 31, 32, 32, 33, 33, 33, 34, 35, 34, 34, 35, 36, 35, 36, 36, 36, 36, 36, 37, 36, 37, 37, 36, 37, 37, 38, 37, 38, 38, 37, 38, 37, 38, 38, 38, 37, 38, 37, 38, 37, 37, 36, 37, 37, 36, 36, 35, 35, 35, 35, 34, 35, 35, 35, 33, 34, 35, 33, 33, 33, 33, 33, 31, 31, 32, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 30, 30, 29, 29, 30, 29, 30, 29, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 31, 31, 31, 31, 32, 32, 32, 33, 33, 32, 33, 33, 34, 34, 34, 35, 34, 35, 36, 35, 35, 34, 36, 36, 36, 37, 37, 37, 37, 37, 38, 37, 38, 38, 37, 37, 37, 38, 38, 38, 38, 37, 37, 37, 37, 38, 37, 37, 36, 37, 36, 36, 36, 36, 36, 37, 36, 34, 35, 34, 35, 33, 34, 33, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 31, 30, 31, 31, 29, 30, 30, 31, 30, 30, 30, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 30, 30, 31, 30, 31, 31, 31, 31, 31, 31, 32, 32, 33, 33, 34, 33, 33, 34, 33, 34, 34, 35, 35, 34, 35, 36, 37, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 37, 38, 38, 38, 39, 37, 38, 38, 37, 38, 38, 37, 38, 37, 37, 38, 37, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35, 34, 33, 33, 33, 34, 32, 33, 32, 32, 32, 31, 32, 31, 31, 31, 31, 31, 30, 29, 30, 30, 30, 29, 29, 29, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 31, 30, 31, 31, 31, 31, 31, 32, 31, 31, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 34, 35, 35, 36, 36, 35, 37, 36, 36, 37, 37, 37, 37, 38, 38, 37, 37, 37, 38, 37, 37, 38, 38, 38, 38, 37, 38, 37, 38, 37, 37, 38, 37, 37, 37, 36, 36, 36, 36, 36, 36, 35, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 32, 31, 32, 31, 31, 31, 30, 31, 30, 29, 30, 29, 29, 30, 29, 29, 30, 29, 29, 30, 30, 30, 30, 29, 30, 30, 30, 30, 29, 31, 31, 31, 30, 31, 31, 31, 31, 32, 32, 32, 33, 32, 33, 34, 34, 35, 34, 35, 34, 36, 35, 36, 36, 36, 36, 36, 36, 37, 37, 38, 37, 37, 37, 38, 37, 38, 39, 37, 37, 37, 39, 38, 38, 38, 38, 36, 37, 37, 36, 36, 37, 37, 37, 36, 36, 36, 35, 35, 35, 34, 34, 34, 34, 33, 33, 32, 33, 33, 32, 32, 32, 31, 32, 31, 31, 31, 30, 30, 31, 30, 30, 30, 31, 30, 30, 30, 30, 29, 30, 30, 30, 30, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 32, 31, 32, 32, 32, 32, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 36, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 38, 38, 37, 37, 38, 38, 37, 37, 38, 38, 37, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35, 34, 34, 34, 35, 33, 33, 33, 32, 32, 32, 32, 31, 31, 32, 31, 31}; 
	short nLen = sizeof(nSampData)/sizeof(short);
	//short *pfSamp = (short *)malloc(nLen*sizeof(short));
	//memset(pfSamp, 0, nLen*sizeof(short));
	float *pfSamp = (float *)malloc(nLen*sizeof(float));
	memset(pfSamp, 0, nLen*sizeof(float));
    unit_ckf_process(nSampData, nLen, pfSamp);
    short i=0;
    for(i=0; i<100; i++){
    	printf("i= %d and  %f \r\n",  i, *(pfSamp+i));
	}
    
    /***/
	unit_ckf_test_case();
	
	system("PAUSE"); 
	return;    
}
#endif
