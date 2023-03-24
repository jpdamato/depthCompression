/*                                                                            */
/*  Sample console unit for simple_Huffman class  v.3.0   (c) 2012 Jan Mojzis */
/*      Notes:                                                                */
/*            x console syntax: <input> <output> <c/d> /compress/decompress   */
/*                                                                            */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "huffman.h"

int main(int argc, char *argv[])
{
    if (argc < 4) { printf("syntax: program <INPUT> <OUTPUT> <c/d>\n"); system("PAUSE"); return 0; }
	
    FILE *finput, *foutput;
    BYTE *input, *output = NULL;
    BYTE op = (BYTE) argv[3][0];
	int delka;
    simple_Huffman *huf = new simple_Huffman();
    // open files
	
    finput = fopen(argv[1], "rb");
    if (!finput) { printf("error upon opening %s\n", argv[1]); system("PAUSE"); return 0; }
    foutput = fopen(argv[2], "wb");
    if (!foutput) { printf("error upon opening %s\n", argv[2]); system("PAUSE"); fclose(finput); return 0; }

    // get the length of input 
    fseek(finput, 0L, SEEK_END);
    delka = ftell(finput);
    fseek(finput, 0L, SEEK_SET);
    
    // read input 
    input = new BYTE[delka];
    fread(input, 1, delka, finput);    
    clock_t before,after;
	
	
    if (op == 'c')
    {
    before = clock();       
    int outsize = huf->Compress(input, delka);
	output = huf->getOutput();
    printf("time: %.2lfs\n", ((double)clock()-before) / 1000.0);
       fwrite(output, outsize, 1, foutput);
       huf->Finalize();
    }
    if (op == 'd')
    {      
    before = clock();
    int outsize = huf->Decompress(input, delka);
	output = huf->getOutput();
    printf("time: %.2lfs\n", ((double)clock()-before) / 1000.0);
       fwrite(output, outsize, 1, foutput);
       huf->Finalize();
    }
	
	// test file compression
	/*
	int outsize;
	//outsize = huf->CompressFile("laclavik_information_extraction.ppt", "laclavik_information_extraction.ppt.pak");
	//outsize = huf->CompressFile("LIC.txt", "LIC.txt.pak");
	//outsize = huf->CompressFile("esej_clanok.rtf", "esej_clanok.rtf.pak");
	//outsize = huf->DecompressFile("LIC.txt.pak", "LIC.txt.dek2");
	//outsize = huf->DecompressFile("esej_clanok.rtf.pak", "esej_clanok.rtf.dek2");
	//outsize = huf->DecompressFile("laclavik_information_extraction.ppt.pak", "laclavik_information_extraction.ppt.dek2");
	*/
    fclose(finput);
    fclose(foutput);
    delete huf;
    delete input;
	system("PAUSE");
    return EXIT_SUCCESS;
}
