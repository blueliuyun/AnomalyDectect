
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

/**
 * def CubatureKalmanFilter():
 * ----------
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
    //self.hx = hx
    //self.fx = fx
    //self.x_mean = x_mean_fn
    //self.z_mean = z_mean_fn
    sCKF.y = 0;
    //self.z = np.array([[None]*self.dim_z]).T
    eye(&sCKF.smZ, 1, nDim_z);
    eye(&sCKF.smS, nDim_z, nDim_z);  // system uncertainty
    eye(&sCKF.smSI, nDim_z, nDim_z); // inverse system uncertainty

    // sigma points transformed through f(x) and h(x)
    // variables for efficiency so we don't recreate every update
    eye(&sCKF.sm_sigmas_f, nDim_x*2, nDim_x);
    eye(&sCKF.sm_sigmas_h, nDim_x*2, nDim_z);

    // these will always be a copy of x,P after predict() is called
    sCKF.saX_prior = sCKF.saX;
    sCKF.smP_prior = sCKF.smP;

    //  these will always be a copy of x,P after update() is called
    sCKF.saX_post = sCKF.saX;
    sCKF.smP_post = sCKF.smP;
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
static /*float*/void eye(MATRIX_F32_STRUCT* s, unsigned short nCol, unsigned short nRow)
{
    // Init MATRIX_F32_STRUCT
    // 1. Check s != NULL
    s->numCols = nCol;
    s->numRows = nRow;
    
    unit_mat_zero_f32(s);
    unit_mat_diagonal_f32(s);
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
 * Performs the predict step of the CKF.
 * Important: this MUST be called before update() is called for the first time.
 * ----------
 */ 
static void unit_ckf_predict()
{

}


/**
 * Creates cubature points for the the specified state and covariance.
 * ----------
 * 返回值类型是数组。
 */ 
static void unit_ckf_spherical_radial_sigmas(MATRIX_F32_STRUCT* saX, MATRIX_F32_STRUCT* smP)
{
    // get Rows of P
    unsigned short nRows = smP->numRows;  // n, _ = P.shape
    float *pData = saX->pData;

    //sigmas = np.empty((2*n, n))  // sigmas 如何返回有效空间内存呢？
    MATRIX_F32_STRUCT sigmas;
    eye(&sigmas, nRows*2, nRows);
    //@2018-12-16 若 P 是单个元素的矩阵, 则矩阵 P 一定是正定矩阵。
    //U = cholesky(smP) * sqrt(n)

}

/**
 * Compute the Cholesky decomposition of a matrix.
 * ----------
 * 返回值。
 */
static void cholesky(MATRIX_F32_STRUCT* smP)
{

}

/**
 * Common code for cholesky() and cho_factor().
 * ----------
 * 返回值。
 */
static void _cholesky(MATRIX_F32_STRUCT* smP)
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
        for (unsigned short k = 0; k < i; k++) {
            unsigned short ki = GetIndexInMatrix(smP, k, i);
            a[ii] = a[ii] - a[ki] * a[ki];
        }

        if (a[ii] < 0) {
            //error , throw std::runtime_error("Matrix is not positive definite.");
            return;
        }

        a[ii] = sqrt(a[ii]);
        for (unsigned short j = i + 1; j < nRows; j++) {
            unsigned short ij = GetIndexInMatrix(smP, i, j);
            for (unsigned short k = 0; k < i; k++) {
                unsigned short ki = GetIndexInMatrix(smP, k, i);
                unsigned short kj = GetIndexInMatrix(smP, k, j);
                a[ij] = a[ij] - a[ki] * a[kj];
            }
            if (a[ij] != 0) a[ij] = a[ij] / a[ii];
        }
    }
    
    // Clear out the lower matrix
    for (unsigned short i = 1; i < nRows; i++) {
        for (unsigned short j = 0; j < i; j++) {
            unsigned short ij = GetIndexInMatrix(smP, i, j);
            a[ij] = 0;
        }
    }
    
    memcpy(smP->pData, a, 4*sizeof(float));
    return ;//Matrix(n, n, a);
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

void main(){
	eye(&sCKF.smP, 2, 2);
	
	*(sCKF.smP.pData) = 1;
	*(sCKF.smP.pData+1) = -2;
	*(sCKF.smP.pData+2) = 2;
	*(sCKF.smP.pData+3) = 5;
	_cholesky(&sCKF.smP);	
} 
