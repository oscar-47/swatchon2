import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import SinglePredict from '../views/SinglePredict.vue'
import BatchPredict from '../views/BatchPredict.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/single',
    name: 'SinglePredict',
    component: SinglePredict
  },
  {
    path: '/batch',
    name: 'BatchPredict',
    component: BatchPredict
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
