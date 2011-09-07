/* 
 * File:   Matrix33.h
 * Author: alexis
 *
 * Created on 12 août 2011, 21:38
 */

#ifndef MATRIX33_H
#define	MATRIX33_H

#include "Vector.h"

class Matrix33 {
	double m_mat[9];

	public:
		Matrix33(void);
		Matrix33(const Matrix33& mat);
		Matrix33(double m00, double m01, double m02, double m10, double m11, double m12,
					double m20, double m21, double m22);
		Matrix33(double m00, double m11, double m22);

		// Create rotation matrices (positive angle -> counterclockwise).  The
		// angle must be in radians, not degrees.
		Matrix33(const Vector& axis, double angle);

		// Assignment.
		Matrix33& operator= (const Matrix33& mat);

		// Data access operators
		double & operator() (int , int);
		double operator() (int , int) const;
		double & operator()(int);
		double operator()(int) const;

		// Create various matrices.
		Matrix33& MakeZero(void);
		Matrix33& MakeIdentity(void);
		Matrix33& MakeDiagonal(double m00, double m11, double m22);
		Matrix33& MakeRotation(const Vector & axis, double angle);

		// Arithmetic operations.
		Matrix33 operator+ (const Matrix33& mat) const;
		Matrix33 operator- (const Matrix33& mat) const;
		Matrix33 operator* (double scalar) const;
		Matrix33 operator/ (double scalar) const;
		Matrix33 operator- () const;

		// Arithmetic updates.
		Matrix33& operator+= (const Matrix33& mat);
		Matrix33& operator-= (const Matrix33& mat);
		Matrix33& operator*= (double scalar);
		Matrix33& operator/= (double scalar);

		// M*vec
		Vector operator* (const Vector& vec) const;
		// vec^T*M
		friend Vector operator* (const Vector& vec, const Matrix33& mat);

		// M^T
		Matrix33 Transpose() const;

		// M*mat
		Matrix33 operator* (const Matrix33& mat) const;

		// M^T*mat
		Matrix33 TransposeTimes(const Matrix33& mat) const;

		// M*mat^T
		Matrix33 TimesTranspose(const Matrix33& mat) const;

		Matrix33 Inverse(const double epsilon = 0.0) const;
		double Determinant(void) const;
		
		// The matrix must be a rotation for these functions to be valid.  The
		// last function uses Gram-Schmidt orthonormalization applied to the
		// columns of the rotation matrix.  The angle must be in radians, not
		// degrees.
		void ExtractAxisAngle(Vector& axis, double& angle) const;

		// Create rotation matrices from Euler angles.
		Matrix33& MakeEulerZXZ(const double z0Angle, const double xAngle, const double z1Angle);

		// Extract Euler angles from rotation matrices.
		enum EulerResult
		{
			// The solution is unique.
			EA_UNIQUE,

			// The solution is not unique.  A sum of angles is constant.
			EA_NOT_UNIQUE_SUM,

			// The solution is not unique.  A difference of angles is constant.
			EA_NOT_UNIQUE_DIF
		};


		// The return values are in the specified ranges:
		//   z0Angle in [-pi,pi], xAngle in [0,pi], z1Angle in [-pi,pi]
		// When the solution is not unique, z1Angle = 0 is returned.  Generally,
		// the set of solutions is
		//   EA_NOT_UNIQUE_SUM:  z1Angle + z0Angle = c
		//   EA_NOT_UNIQUE_DIF:  z1Angle - z0Angle = c
		// for some angle c.
		EulerResult ExtractEulerZXZ(double& z0Angle, double& xAngle, double& z1Angle) const;
};
#endif	/* MATRIX33_H */

