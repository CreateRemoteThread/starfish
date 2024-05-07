// based off https://raw.githubusercontent.com/RedPitaya/RedPitaya/master/Examples/C/axi.c

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "rp.h"

#define DATA_SIZE 128000

int main(int argc, char **argv)
{
    int dsize = DATA_SIZE;
    uint32_t dec = 1;
    if (argc >= 3){
        dsize = atoi(argv[1]);
        dec = atoi(argv[2]);
    }

    /* Print error, if rp_Init() function failed */
    if (rp_InitReset(false) != RP_OK) {
        fprintf(stderr, "Rp api init failed!\n");
        return -1;
    }

    uint32_t g_adc_axi_start,g_adc_axi_size;
    rp_AcqAxiGetMemoryRegion(&g_adc_axi_start,&g_adc_axi_size);

    printf("Reserved memory start 0x%X size 0x%X\n",g_adc_axi_start,g_adc_axi_size);

    // rp_AcqResetFpga();

    if (rp_AcqAxiSetDecimationFactor(dec) != RP_OK) {
        fprintf(stderr, "rp_AcqAxiSetDecimationFactor failed!\n");
        return -1;
    }
    if (rp_AcqAxiSetTriggerDelay(RP_CH_1, dsize  )  != RP_OK) {
        fprintf(stderr, "rp_AcqAxiSetTriggerDelay RP_CH_1 failed!\n");
        return -1;
    }
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, g_adc_axi_start, dsize) != RP_OK) {
        fprintf(stderr, "rp_AcqAxiSetBuffer RP_CH_1 failed!\n");
        return -1;
    }
    if (rp_AcqAxiEnable(RP_CH_1, true)) {
        fprintf(stderr, "rp_AcqAxiEnable RP_CH_1 failed!\n");
        return -1;
    }

    // 10mV
    rp_AcqSetTriggerLevel(RP_T_CH_1,0.01);

    if (rp_AcqStart() != RP_OK) {
        fprintf(stderr, "rp_AcqStart failed!\n");
        return -1;
    }

    rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
    rp_acq_trig_state_t state = RP_TRIG_STATE_TRIGGERED;

    while(1){
        rp_AcqGetTriggerState(&state);
        if(state == RP_TRIG_STATE_TRIGGERED){
            // tiny skip only :)
            sleep(0.001);
            break;
        }
    }

    bool fillState = false;
    while (!fillState) {
        if (rp_AcqAxiGetBufferFillState(RP_CH_1, &fillState) != RP_OK) {
            fprintf(stderr, "rp_AcqAxiGetBufferFillState RP_CH_1 failed!\n");
            return -1;
        }
    }
    rp_AcqStop();

    uint32_t posChA;
    rp_AcqAxiGetWritePointerAtTrig(RP_CH_1,&posChA);

    fprintf(stderr,"Tr pos1: 0x%X\n",posChA);

    int16_t *buff1 = (int16_t *)malloc(dsize * sizeof(int16_t));

    uint32_t size1 = dsize;
    rp_AcqAxiGetDataRaw(RP_CH_1, posChA, &size1, buff1);

    printf("acquired ok");
    /*
    for (int i = 0; i < dsize; i++) {
        printf("[%d]\t%d\t%d\n",i,buff1[i], buff2[i]);
    }
    */
    for(int i = 0;i < (dsize - 2000);i++)
    {
      if(abs(buff1[i]) > 160)
      {
        printf("%d : %d\n",i,buff1[i]);
        i += 2000;
      }
      // z = buff1[i] + 4;
    }

    /* Releasing resources */
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    free(buff1);
    return 0;
}
