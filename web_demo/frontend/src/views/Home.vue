<template>
  <div>
    <!-- Hero Section -->
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 80px 0; margin: -24px -24px 40px -24px; text-align: center; color: white;">
      <h1 style="font-size: 48px; margin-bottom: 20px; font-weight: 300;">
        <span style="margin-right: 16px;">🧵</span>
        面料智能分类系统
      </h1>
      <p style="font-size: 20px; margin-bottom: 40px; opacity: 0.9;">
        基于深度学习的纺织品自动识别与分类平台
      </p>
      <a-space size="large">
        <a-button type="primary" size="large" @click="$router.push('/single')" style="height: 50px; padding: 0 30px; font-size: 16px;">
          <template #icon>
            <picture-outlined />
          </template>
          开始单图识别
        </a-button>
        <a-button size="large" @click="$router.push('/batch')" style="height: 50px; padding: 0 30px; font-size: 16px; background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.4); color: white;">
          <template #icon>
            <folder-open-outlined />
          </template>
          批量识别
        </a-button>
      </a-space>
    </div>

    <!-- Features Section -->
    <a-row :gutter="[32, 32]" style="margin-bottom: 60px;">
      <a-col :xs="24" :md="8">
        <a-card hoverable style="text-align: center; height: 100%;">
          <template #cover>
            <div style="padding: 40px; font-size: 64px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
              🎯
            </div>
          </template>
          <a-card-meta title="高精度识别" description="基于ResNet50深度学习模型，识别准确率超过85%，支持多种面料类型的精确分类" />
        </a-card>
      </a-col>
      <a-col :xs="24" :md="8">
        <a-card hoverable style="text-align: center; height: 100%;">
          <template #cover>
            <div style="padding: 40px; font-size: 64px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
              ⚡
            </div>
          </template>
          <a-card-meta title="快速响应" description="GPU加速推理，单张图片识别时间小于1秒，支持批量处理提升工作效率" />
        </a-card>
      </a-col>
      <a-col :xs="24" :md="8">
        <a-card hoverable style="text-align: center; height: 100%;">
          <template #cover>
            <div style="padding: 40px; font-size: 64px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
              🔧
            </div>
          </template>
          <a-card-meta title="专业易用" description="直观的Web界面，支持拖拽上传，实时预览结果，无需专业技术背景即可使用" />
        </a-card>
      </a-col>
    </a-row>

    <!-- Models Section -->
    <a-card title="可用模型" style="margin-bottom: 40px;">
      <template #extra>
        <a-tag color="processing">{{ models.length }} 个模型可用</a-tag>
      </template>
      
      <a-spin :spinning="loading">
        <a-row :gutter="[24, 24]" v-if="models.length > 0">
          <a-col :xs="24" :lg="8" v-for="model in models" :key="model.id">
            <a-card size="small" hoverable>
              <template #title>
                <a-space>
                  <a-tag :color="model.type === 'binary' ? 'blue' : 'green'">
                    {{ model.type === 'binary' ? '二分类' : '多分类' }}
                  </a-tag>
                  {{ model.name }}
                </a-space>
              </template>
              <p style="color: #666; margin-bottom: 16px;">{{ model.description }}</p>
              <a-space>
                <a-button type="primary" size="small" @click="useModel(model, 'single')">
                  单图识别
                </a-button>
                <a-button size="small" @click="useModel(model, 'batch')">
                  批量识别
                </a-button>
              </a-space>
            </a-card>
          </a-col>
        </a-row>
        
        <a-empty v-else description="暂无可用模型" />
      </a-spin>
    </a-card>

    <!-- Stats Section -->
    <a-row :gutter="[32, 32]">
      <a-col :xs="24" :md="6">
        <a-statistic title="支持面料类型" :value="19" suffix="种" />
      </a-col>
      <a-col :xs="24" :md="6">
        <a-statistic title="训练样本数量" :value="5000" suffix="+" />
      </a-col>
      <a-col :xs="24" :md="6">
        <a-statistic title="平均识别准确率" :value="85.2" suffix="%" />
      </a-col>
      <a-col :xs="24" :md="6">
        <a-statistic title="平均响应时间" :value="0.8" suffix="秒" />
      </a-col>
    </a-row>
  </div>
</template>

<script>
import { PictureOutlined, FolderOpenOutlined } from '@ant-design/icons-vue'
import axios from 'axios'

export default {
  name: 'Home',
  components: {
    PictureOutlined,
    FolderOpenOutlined
  },
  data() {
    return {
      models: [],
      loading: false
    }
  },
  mounted() {
    this.loadModels()
  },
  methods: {
    async loadModels() {
      this.loading = true
      try {
        const response = await axios.get('/api/models')
        this.models = response.data.models
      } catch (error) {
        this.$message.error('加载模型失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    useModel(model, type) {
      this.$router.push({
        path: `/${type}`,
        query: { model: model.id }
      })
    }
  }
}
</script>
