<template>
  <div>
    <a-page-header
      title="单图识别"
      sub-title="上传单张面料图片进行智能分类识别"
      @back="$router.go(-1)"
    />

    <a-row :gutter="32">
      <a-col :xs="24" :lg="12">
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
            :show-upload-list="false"
            accept="image/*"
            @change="handleUpload"
          >
            <div v-if="!imageUrl" style="padding: 40px;">
              <p class="ant-upload-drag-icon">
                <inbox-outlined style="font-size: 48px; color: #1890ff;" />
              </p>
              <p class="ant-upload-text">点击或拖拽图片到此区域上传</p>
              <p class="ant-upload-hint">支持 JPG、PNG、BMP、WEBP 格式</p>
            </div>
            <div v-else style="position: relative;">
              <img :src="imageUrl" style="max-width: 100%; max-height: 300px; object-fit: contain;" />
              <a-button
                type="primary"
                danger
                size="small"
                style="position: absolute; top: 8px; right: 8px;"
                @click.stop="clearImage"
              >
                <template #icon>
                  <delete-outlined />
                </template>
              </a-button>
            </div>
          </a-upload-dragger>

          <div style="margin-top: 16px; text-align: center;">
            <a-button
              type="primary"
              size="large"
              :loading="predicting"
              :disabled="!imageUrl || !selectedModel"
              @click="predict"
            >
              <template #icon>
                <thunderbolt-outlined />
              </template>
              开始识别
            </a-button>
          </div>
        </a-card>
      </a-col>

      <a-col :xs="24" :lg="12">
        <a-card title="识别结果">
          <div v-if="!result && !predicting" style="text-align: center; padding: 60px 20px; color: #999;">
            <picture-outlined style="font-size: 64px; margin-bottom: 16px;" />
            <p>请上传图片并选择模型进行识别</p>
          </div>

          <a-spin :spinning="predicting" style="width: 100%;">
            <div v-if="result">
              <a-alert
                :type="result.predictions[0].confidence > 0.8 ? 'success' : result.predictions[0].confidence > 0.6 ? 'warning' : 'info'"
                :message="`识别结果: ${result.predictions[0].class}`"
                :description="`置信度: ${(result.predictions[0].confidence * 100).toFixed(1)}%`"
                show-icon
                style="margin-bottom: 24px;"
              />

              <h4 style="margin-bottom: 16px;">详细预测结果</h4>
              <a-list
                :data-source="result.predictions"
                size="small"
              >
                <template #renderItem="{ item, index }">
                  <a-list-item>
                    <a-list-item-meta>
                      <template #title>
                        <a-space>
                          <a-tag :color="index === 0 ? 'blue' : 'default'">
                            #{{ item.rank }}
                          </a-tag>
                          {{ item.class }}
                        </a-space>
                      </template>
                      <template #description>
                        <a-progress
                          :percent="item.confidence * 100"
                          :stroke-color="index === 0 ? '#1890ff' : '#d9d9d9'"
                          size="small"
                        />
                      </template>
                    </a-list-item-meta>
                  </a-list-item>
                </template>
              </a-list>

              <a-divider />
              
              <a-descriptions title="识别信息" size="small" :column="1">
                <a-descriptions-item label="使用模型">{{ result.model.name }}</a-descriptions-item>
                <a-descriptions-item label="模型类型">{{ result.model.type === 'binary' ? '二分类' : '多分类' }}</a-descriptions-item>
                <a-descriptions-item label="识别时间">{{ new Date(result.timestamp * 1000).toLocaleString() }}</a-descriptions-item>
              </a-descriptions>
            </div>
          </a-spin>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script>
import { InboxOutlined, DeleteOutlined, ThunderboltOutlined, PictureOutlined } from '@ant-design/icons-vue'
import axios from 'axios'

export default {
  name: 'SinglePredict',
  components: {
    InboxOutlined,
    DeleteOutlined,
    ThunderboltOutlined,
    PictureOutlined
  },
  data() {
    return {
      models: [],
      selectedModel: null,
      fileList: [],
      imageUrl: null,
      imageData: null,
      predicting: false,
      result: null
    }
  },
  computed: {
    currentModelInfo() {
      return this.models.find(m => m.id === this.selectedModel)
    }
  },
  mounted() {
    this.loadModels()
    // Check if model is pre-selected from route query
    if (this.$route.query.model) {
      this.selectedModel = this.$route.query.model
    }
  },
  methods: {
    async loadModels() {
      try {
        const response = await axios.get('/api/models')
        this.models = response.data.models
        
        // Auto-select first model if none selected
        if (!this.selectedModel && this.models.length > 0) {
          this.selectedModel = this.models[0].id
        }
      } catch (error) {
        this.$message.error('加载模型失败: ' + error.message)
      }
    },
    
    onModelChange() {
      this.result = null
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
      
      return false // Prevent auto upload
    },
    
    handleUpload(info) {
      const file = info.file
      if (file.status !== 'removed') {
        this.getBase64(file.originFileObj || file, (imageUrl) => {
          this.imageUrl = imageUrl
          this.imageData = imageUrl
          this.result = null
        })
      }
    },
    
    getBase64(file, callback) {
      const reader = new FileReader()
      reader.addEventListener('load', () => callback(reader.result))
      reader.readAsDataURL(file)
    },
    
    clearImage() {
      this.imageUrl = null
      this.imageData = null
      this.fileList = []
      this.result = null
    },
    
    async predict() {
      if (!this.imageData || !this.selectedModel) {
        this.$message.warning('请上传图片并选择模型')
        return
      }
      
      this.predicting = true
      try {
        const response = await axios.post('/api/predict', {
          model_id: this.selectedModel,
          image: this.imageData
        })
        
        this.result = response.data
        this.$message.success('识别完成!')
      } catch (error) {
        this.$message.error('识别失败: ' + (error.response?.data?.error || error.message))
      } finally {
        this.predicting = false
      }
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
