<template>
  <div>
    <a-page-header
      title="批量识别"
      sub-title="上传多张面料图片进行批量智能分类识别"
      @back="$router.go(-1)"
    />

    <a-row :gutter="32">
      <a-col :xs="24" :lg="8">
        <a-card title="选择模型" style="margin-bottom: 24px;">
          <a-select
            v-model:value="selectedModel"
            placeholder="请选择识别模型"
            style="width: 100%"
            @change="onModelChange"
          >
            <a-select-option v-for="model in models" :key="model.id" :value="model.id">
              <a-space>
                <a-tag :color="model.type === 'binary' ? 'blue' : 'green'" size="small">
                  {{ model.type === 'binary' ? '二分类' : '多分类' }}
                </a-tag>
                {{ model.name }}
              </a-space>
            </a-select-option>
          </a-select>
          <p v-if="currentModelInfo" style="margin-top: 8px; color: #666; font-size: 14px;">
            {{ currentModelInfo.description }}
          </p>
        </a-card>

        <a-card title="上传图片">
          <a-upload-dragger
            v-model:fileList="fileList"
            :before-upload="beforeUpload"
            multiple
            accept="image/*"
            @change="handleUpload"
          >
            <div style="padding: 40px;">
              <p class="ant-upload-drag-icon">
                <inbox-outlined style="font-size: 48px; color: #1890ff;" />
              </p>
              <p class="ant-upload-text">点击或拖拽图片到此区域上传</p>
              <p class="ant-upload-hint">支持多选，JPG、PNG、BMP、WEBP 格式</p>
            </div>
          </a-upload-dragger>

          <div v-if="imageDataList.length > 0" style="margin-top: 16px;">
            <a-space>
              <a-tag color="blue">已选择 {{ imageDataList.length }} 张图片</a-tag>
              <a-button size="small" @click="clearImages">清空</a-button>
            </a-space>
          </div>

          <div style="margin-top: 16px; text-align: center;">
            <a-button
              type="primary"
              size="large"
              :loading="predicting"
              :disabled="imageDataList.length === 0 || !selectedModel"
              @click="batchPredict"
            >
              <template #icon>
                <thunderbolt-outlined />
              </template>
              批量识别 ({{ imageDataList.length }})
            </a-button>
          </div>
        </a-card>
      </a-col>

      <a-col :xs="24" :lg="16">
        <a-card title="识别结果">
          <template #extra v-if="results.length > 0">
            <a-space>
              <a-button size="small" @click="exportResults">
                <template #icon>
                  <download-outlined />
                </template>
                导出结果
              </a-button>
              <a-button size="small" @click="clearResults">清空结果</a-button>
            </a-space>
          </template>

          <div v-if="results.length === 0 && !predicting" style="text-align: center; padding: 60px 20px; color: #999;">
            <folder-open-outlined style="font-size: 64px; margin-bottom: 16px;" />
            <p>请上传图片并选择模型进行批量识别</p>
          </div>

          <a-spin :spinning="predicting" style="width: 100%;">
            <div v-if="predicting" style="text-align: center; padding: 40px;">
              <p>正在处理 {{ imageDataList.length }} 张图片...</p>
              <a-progress :percent="Math.round((processedCount / imageDataList.length) * 100)" />
            </div>

            <div v-if="results.length > 0">
              <a-row :gutter="[16, 16]">
                <a-col :xs="24" :sm="12" :lg="8" v-for="(result, index) in results" :key="index">
                  <a-card size="small" hoverable>
                    <template #cover>
                      <div style="height: 200px; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #f5f5f5;">
                        <img
                          :src="imageDataList[result.index]"
                          style="max-width: 100%; max-height: 100%; object-fit: contain;"
                        />
                      </div>
                    </template>
                    
                    <div v-if="result.success">
                      <a-space direction="vertical" style="width: 100%;">
                        <a-tag :color="result.prediction.confidence > 0.8 ? 'green' : result.prediction.confidence > 0.6 ? 'orange' : 'blue'">
                          {{ result.prediction.class }}
                        </a-tag>
                        <a-progress
                          :percent="result.prediction.confidence * 100"
                          size="small"
                          :stroke-color="result.prediction.confidence > 0.8 ? '#52c41a' : result.prediction.confidence > 0.6 ? '#faad14' : '#1890ff'"
                        />
                        <small style="color: #666;">
                          置信度: {{ (result.prediction.confidence * 100).toFixed(1) }}%
                        </small>
                      </a-space>
                    </div>
                    
                    <div v-else>
                      <a-alert type="error" :message="result.error" size="small" />
                    </div>
                  </a-card>
                </a-col>
              </a-row>

              <a-divider />
              
              <a-row :gutter="32">
                <a-col :xs="24" :md="6">
                  <a-statistic title="总计" :value="results.length" suffix="张" />
                </a-col>
                <a-col :xs="24" :md="6">
                  <a-statistic title="成功" :value="successCount" suffix="张" />
                </a-col>
                <a-col :xs="24" :md="6">
                  <a-statistic title="失败" :value="failCount" suffix="张" />
                </a-col>
                <a-col :xs="24" :md="6">
                  <a-statistic title="成功率" :value="successRate" suffix="%" />
                </a-col>
              </a-row>
            </div>
          </a-spin>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script>
import { InboxOutlined, ThunderboltOutlined, FolderOpenOutlined, DownloadOutlined } from '@ant-design/icons-vue'
import axios from 'axios'

export default {
  name: 'BatchPredict',
  components: {
    InboxOutlined,
    ThunderboltOutlined,
    FolderOpenOutlined,
    DownloadOutlined
  },
  data() {
    return {
      models: [],
      selectedModel: null,
      fileList: [],
      imageDataList: [],
      predicting: false,
      results: [],
      processedCount: 0
    }
  },
  computed: {
    currentModelInfo() {
      return this.models.find(m => m.id === this.selectedModel)
    },
    successCount() {
      return this.results.filter(r => r.success).length
    },
    failCount() {
      return this.results.filter(r => !r.success).length
    },
    successRate() {
      return this.results.length > 0 ? Math.round((this.successCount / this.results.length) * 100) : 0
    }
  },
  mounted() {
    this.loadModels()
    if (this.$route.query.model) {
      this.selectedModel = this.$route.query.model
    }
  },
  methods: {
    async loadModels() {
      try {
        const response = await axios.get('/api/models')
        this.models = response.data.models
        
        if (!this.selectedModel && this.models.length > 0) {
          this.selectedModel = this.models[0].id
        }
      } catch (error) {
        this.$message.error('加载模型失败: ' + error.message)
      }
    },
    
    onModelChange() {
      this.results = []
    },
    
    beforeUpload(file) {
      const isImage = file.type.startsWith('image/')
      if (!isImage) {
        this.$message.error('只能上传图片文件!')
        return false
      }
      
      const isLt10M = file.size / 1024 / 1024 < 10
      if (!isLt10M) {
        this.$message.error('图片大小不能超过 10MB!')
        return false
      }
      
      return false
    },
    
    handleUpload(info) {
      const files = info.fileList.map(f => f.originFileObj || f)
      this.processFiles(files)
    },
    
    processFiles(files) {
      this.imageDataList = []
      const promises = files.map(file => {
        return new Promise((resolve) => {
          const reader = new FileReader()
          reader.onload = (e) => resolve(e.target.result)
          reader.readAsDataURL(file)
        })
      })
      
      Promise.all(promises).then(results => {
        this.imageDataList = results
        this.results = []
      })
    },
    
    clearImages() {
      this.fileList = []
      this.imageDataList = []
      this.results = []
    },
    
    clearResults() {
      this.results = []
    },
    
    async batchPredict() {
      if (this.imageDataList.length === 0 || !this.selectedModel) {
        this.$message.warning('请上传图片并选择模型')
        return
      }
      
      this.predicting = true
      this.processedCount = 0
      
      try {
        const response = await axios.post('/api/batch_predict', {
          model_id: this.selectedModel,
          images: this.imageDataList
        })
        
        this.results = response.data.results
        this.processedCount = this.imageDataList.length
        
        this.$message.success(`批量识别完成! 成功: ${this.successCount}, 失败: ${this.failCount}`)
      } catch (error) {
        this.$message.error('批量识别失败: ' + (error.response?.data?.error || error.message))
      } finally {
        this.predicting = false
      }
    },
    
    exportResults() {
      if (this.results.length === 0) return
      
      const csvContent = [
        ['序号', '预测类别', '置信度', '状态'],
        ...this.results.map((result, index) => [
          index + 1,
          result.success ? result.prediction.class : '识别失败',
          result.success ? (result.prediction.confidence * 100).toFixed(2) + '%' : '-',
          result.success ? '成功' : result.error
        ])
      ].map(row => row.join(',')).join('\n')
      
      const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = `批量识别结果_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`
      link.click()
    }
  }
}
</script>

<style scoped>
.ant-upload-dragger {
  background: #fafafa !important;
}

.ant-upload-dragger:hover {
  border-color: #1890ff !important;
}
</style>
