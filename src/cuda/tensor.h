#ifndef _TENSOR_H_
#define _TENSOR_H_

typedef struct {
	float xx;
	float xy;
	float xz;
	float yy;
	float yz;
	float zz;
} symtensor3 ;

typedef struct {
	float xx;
	float xy;
	float xz;
	float xw;
	float yy;
	float yz;
	float yw;
	float zz;
	float zw;
	float ww;
} symtensor4 ;

#endif
