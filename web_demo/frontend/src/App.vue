<template>
  <div id="app">
    <a-layout style="min-height: 100vh">
      <a-layout-header style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0 50px;">
        <div style="display: flex; align-items: center; height: 64px;">
          <div style="color: white; font-size: 24px; font-weight: bold; margin-right: 50px;">
            <span style="margin-right: 8px;">🧵</span>
            面料智能分类系统
          </div>
          <a-menu
            mode="horizontal"
            :selected-keys="[currentRoute]"
            style="background: transparent; border-bottom: none; flex: 1;"
            theme="dark"
          >
            <a-menu-item key="/" @click="$router.push('/')">
              <template #icon>
                <home-outlined />
              </template>
              首页
            </a-menu-item>
            <a-menu-item key="/single" @click="$router.push('/single')">
              <template #icon>
                <picture-outlined />
              </template>
              单图识别
            </a-menu-item>
            <a-menu-item key="/batch" @click="$router.push('/batch')">
              <template #icon>
                <folder-open-outlined />
              </template>
              批量识别
            </a-menu-item>
          </a-menu>
        </div>
      </a-layout-header>
      
      <a-layout-content style="padding: 24px; background: #f0f2f5;">
        <router-view />
      </a-layout-content>
      
      <a-layout-footer style="text-align: center; background: #001529; color: rgba(255,255,255,0.65);">
        面料智能分类系统 ©2024 基于深度学习的纺织品识别技术
      </a-layout-footer>
    </a-layout>
  </div>
</template>

<script>
import { HomeOutlined, PictureOutlined, FolderOpenOutlined } from '@ant-design/icons-vue'

export default {
  name: 'App',
  components: {
    HomeOutlined,
    PictureOutlined,
    FolderOpenOutlined
  },
  computed: {
    currentRoute() {
      return this.$route.path
    }
  }
}
</script>

<style>
#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
}

.ant-layout-header .ant-menu-horizontal {
  line-height: 64px;
}

.ant-menu-horizontal > .ant-menu-item {
  border-bottom: 2px solid transparent !important;
}

.ant-menu-horizontal > .ant-menu-item:hover,
.ant-menu-horizontal > .ant-menu-item-selected {
  border-bottom-color: #fff !important;
  color: #fff !important;
}
</style>
