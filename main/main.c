#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_task_wdt.h"

#define EXAMPLE_ADC_UNIT             ADC_UNIT_1
#define EXAMPLE_ADC_CHANNEL          ADC_CHANNEL_3 // GPIO 1 on ESP32-S3
#define EXAMPLE_ADC_ATTEN            ADC_ATTEN_DB_12
#define EXAMPLE_ADC_BITWIDTH         ADC_BITWIDTH_12

#define SAMPLE_FREQ_HZ               1000  // 2 kHz sampling rate
#define READ_LEN                     16    // 32 bytes = 8 samples per frame

static TaskHandle_t s_adc_task_handle = NULL;

// DMA Event Callback: Notifies our dedicated task when a conversion frame is ready
static bool IRAM_ATTR adc_coex_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data)
{
    BaseType_t mustYield = pdFALSE;
    if (s_adc_task_handle != NULL) {
        vTaskNotifyGiveFromISR(s_adc_task_handle, &mustYield);
    }
    return mustYield;
}

// The dedicated processing task pinned entirely to Core 1
void adc_plot_task(void *pvParameters)
{
    // Capture our own task handle so the ISR callback can notify us
    s_adc_task_handle = xTaskGetCurrentTaskHandle();

    // 1. Initialize Continuous Mode Driver Handle
    adc_continuous_handle_t adc_handle = NULL;
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 1024,
        .conv_frame_size = READ_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &adc_handle));

    // 2. Configure the ADC Channel Patterns
    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = SAMPLE_FREQ_HZ,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1, 
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2, 
        .pattern_num = 1,
    };

    adc_digi_pattern_config_t adc_pattern = {
        .atten = EXAMPLE_ADC_ATTEN,
        .channel = EXAMPLE_ADC_CHANNEL,
        .unit = EXAMPLE_ADC_UNIT,
        .bit_width = EXAMPLE_ADC_BITWIDTH,
    };
    dig_cfg.adc_pattern = &adc_pattern;
    
    ESP_ERROR_CHECK(adc_continuous_config(adc_handle, &dig_cfg));

    // 3. Register Callback and Start ADC
    adc_continuous_evt_cbs_t cbs = {
        .on_conv_done = adc_coex_cb,
    };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(adc_handle, &cbs, NULL));
    ESP_ERROR_CHECK(adc_continuous_start(adc_handle));

    // 4. Register this specific task with the Core 1 Watchdog (TWDT)
    ESP_ERROR_CHECK(esp_task_wdt_add(s_adc_task_handle));

    uint8_t result[READ_LEN] = {0};
    uint32_t ret_num = 0;

    while (1) {
        // Wait until the DMA fills a frame (Blocks efficiently without burning CPU)
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        esp_err_t ret = adc_continuous_read(adc_handle, result, READ_LEN, &ret_num, 0);
        if (ret == ESP_OK) {
            uint32_t samples_count = ret_num / SOC_ADC_DIGI_RESULT_BYTES;
            adc_continuous_data_t parsed_data[samples_count];
            uint32_t num_parsed = 0;

            if (adc_continuous_parse_data(adc_handle, result, ret_num, parsed_data, &num_parsed) == ESP_OK) {
                for (int i = 0; i < num_parsed; i++) {
                    printf("%d\n", (int)parsed_data[i].raw_data);
                }
            }
        }

        // Feed the watchdog for this task specifically
        esp_task_wdt_reset();

        // Yield momentarily to let IDLE1 execute on Core 1
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    // Clean up if the loop ever breaks (unreachable here)
    esp_task_wdt_delete(s_adc_task_handle);
    adc_continuous_stop(adc_handle);
    adc_continuous_deinit(adc_handle);
    vTaskDelete(NULL);
}

void app_main(void)
{
    // Create the ADC plotting task
    // Parameters: Task Function, Name, Stack Size (bytes), Params, Priority, Handle, Core ID (1)
    xTaskCreatePinnedToCore(
        adc_plot_task, 
        "adc_plot_task", 
        4096, 
        NULL, 
        5, 
        NULL, 
        1
    );

    // app_main runs on Core 0. Since its job is done, we can just let it delete itself.
    // This removes it from the task scheduler completely.
    vTaskDelete(NULL);
}
